package com.metawebthree.payment.application;

import com.metawebthree.payment.domain.model.CryptoPrice;
import com.metawebthree.payment.infrastructure.persistence.mapper.CryptoPriceRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
@RequiredArgsConstructor
@Slf4j
public class PriceEngineServiceImpl {
    
    private final CryptoPriceRepository cryptoPriceRepository;
    private final ExternalPriceServiceImpl externalPriceService;
    
    @Value("${payment.price-engine.update-interval:5}")
    private int updateInterval;
    
    private final Map<String, CryptoPrice> priceCache = new ConcurrentHashMap<>();
    
    @Cacheable(value = "cryptoPrices", key = "#symbol")
    public CryptoPrice getCurrentPrice(String symbol) {
        CryptoPrice cachedPrice = priceCache.get(symbol);
        if (cachedPrice != null && isPriceValid(cachedPrice)) {
            return cachedPrice;
        }
        CryptoPrice dbPrice = cryptoPriceRepository.findFirstBySymbolOrderByTimestampDesc(symbol);
        if (dbPrice != null) {
            return dbPrice;
        }
        return fetchAndSavePrice(symbol);
    }
    
    public BigDecimal getWeightedAveragePrice(String baseCurrency, String quoteCurrency) {
        List<CryptoPrice> prices = cryptoPriceRepository.findByBaseCurrencyAndQuoteCurrency(baseCurrency, quoteCurrency);
        
        if (prices.isEmpty()) {
            throw new RuntimeException("No price data available for " + baseCurrency + "-" + quoteCurrency);
        }
        BigDecimal totalWeight = BigDecimal.ZERO;
        BigDecimal weightedSum = BigDecimal.ZERO;
        Map<String, BigDecimal> weights = Map.of(
            "binance", new BigDecimal("0.4"),
            "coinbase", new BigDecimal("0.3"),
            "okx", new BigDecimal("0.3")
        );
        for (CryptoPrice price : prices) {
            BigDecimal weight = weights.getOrDefault(price.getSource(), new BigDecimal("0.1"));
            weightedSum = weightedSum.add(price.getPrice().multiply(weight));
            totalWeight = totalWeight.add(weight);
        }
        
        return weightedSum.divide(totalWeight, 8, RoundingMode.HALF_UP);
    }
    
    public BigDecimal calculateExchangeRate(String fromCurrency, String toCurrency, BigDecimal amount) {
        if (fromCurrency.equals(toCurrency)) {
            return BigDecimal.ONE;
        }
        BigDecimal rate = getWeightedAveragePrice(fromCurrency, toCurrency);
        return amount.multiply(rate).setScale(8, RoundingMode.HALF_UP);
    }
    
    @Scheduled(fixedRate = 5000) // 5 seconds
    public void updatePrices() {
        log.info("Starting price update...");
        
        String[] symbols = {"BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD"};
        
        for (String symbol : symbols) {
            try {
                fetchAndSavePrice(symbol);
                log.debug("Updated price for {}", symbol);
            } catch (Exception e) {
                log.error("Failed to update price for {}: {}", symbol, e.getMessage());
            }
        }
    }
    
    private CryptoPrice fetchAndSavePrice(String symbol) {
        try {
            CryptoPrice price = externalPriceService.fetchPrice(symbol);
            if (price != null) {
                cryptoPriceRepository.insert(price);
                priceCache.put(symbol, price);
                return price;
            }
        } catch (Exception e) {
            log.error("Failed to fetch price for {}: {}", symbol, e.getMessage());
        }
        CryptoPrice cachedPrice = priceCache.get(symbol);
        if (cachedPrice != null) {
            return cachedPrice;
        }
        throw new RuntimeException("Unable to fetch price for " + symbol);
    }
    
    private boolean isPriceValid(CryptoPrice price) {
        return price.getTimestamp().isAfter(LocalDateTime.now().minusMinutes(5));
    }
    
    public BigDecimal getPriceChangePercent(String symbol, int hours) {
        LocalDateTime startTime = LocalDateTime.now().minusHours(hours);
        List<CryptoPrice> prices = cryptoPriceRepository.findBySymbolAndTimestampAfter(symbol, startTime);
        
        if (prices.size() < 2) {
            return BigDecimal.ZERO;
        }
        
        CryptoPrice latest = prices.get(0);
        CryptoPrice oldest = prices.get(prices.size() - 1);
        
        BigDecimal change = latest.getPrice().subtract(oldest.getPrice());
        return change.divide(oldest.getPrice(), 4, RoundingMode.HALF_UP)
                .multiply(new BigDecimal("100"));
    }
} 