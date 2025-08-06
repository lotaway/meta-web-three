package com.metawebthree.service;

import com.metawebthree.entity.CryptoPrice;
import com.metawebthree.repository.CryptoPriceRepository;
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
public class PriceEngineService {
    
    private final CryptoPriceRepository cryptoPriceRepository;
    private final ExternalPriceService externalPriceService;
    
    @Value("${payment.price-engine.update-interval:5}")
    private int updateInterval;
    
    // 内存缓存，避免频繁数据库查询
    private final Map<String, CryptoPrice> priceCache = new ConcurrentHashMap<>();
    
    /**
     * 获取实时价格
     */
    @Cacheable(value = "cryptoPrices", key = "#symbol")
    public CryptoPrice getCurrentPrice(String symbol) {
        // 先从缓存获取
        CryptoPrice cachedPrice = priceCache.get(symbol);
        if (cachedPrice != null && isPriceValid(cachedPrice)) {
            return cachedPrice;
        }
        
        // 从数据库获取最新价格
        return cryptoPriceRepository.findFirstBySymbolOrderByTimestampDesc(symbol)
                .orElseGet(() -> fetchAndSavePrice(symbol));
    }
    
    /**
     * 获取加权平均价格
     */
    public BigDecimal getWeightedAveragePrice(String baseCurrency, String quoteCurrency) {
        List<CryptoPrice> prices = cryptoPriceRepository.findByBaseCurrencyAndQuoteCurrency(baseCurrency, quoteCurrency);
        
        if (prices.isEmpty()) {
            throw new RuntimeException("No price data available for " + baseCurrency + "-" + quoteCurrency);
        }
        
        BigDecimal totalWeight = BigDecimal.ZERO;
        BigDecimal weightedSum = BigDecimal.ZERO;
        
        // 权重配置
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
    
    /**
     * 计算兑换汇率
     */
    public BigDecimal calculateExchangeRate(String fromCurrency, String toCurrency, BigDecimal amount) {
        if (fromCurrency.equals(toCurrency)) {
            return BigDecimal.ONE;
        }
        
        // 获取汇率
        BigDecimal rate = getWeightedAveragePrice(fromCurrency, toCurrency);
        
        // 计算兑换金额
        return amount.multiply(rate).setScale(8, RoundingMode.HALF_UP);
    }
    
    /**
     * 定时更新价格数据
     */
    @Scheduled(fixedRate = 5000) // 每5秒更新一次
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
    
    /**
     * 从外部API获取价格并保存
     */
    private CryptoPrice fetchAndSavePrice(String symbol) {
        try {
            CryptoPrice price = externalPriceService.fetchPrice(symbol);
            if (price != null) {
                CryptoPrice savedPrice = cryptoPriceRepository.save(price);
                priceCache.put(symbol, savedPrice);
                return savedPrice;
            }
        } catch (Exception e) {
            log.error("Failed to fetch price for {}: {}", symbol, e.getMessage());
        }
        
        // 如果获取失败，返回缓存中的旧价格或抛出异常
        CryptoPrice cachedPrice = priceCache.get(symbol);
        if (cachedPrice != null) {
            return cachedPrice;
        }
        
        throw new RuntimeException("Unable to fetch price for " + symbol);
    }
    
    /**
     * 检查价格是否有效（不超过5分钟）
     */
    private boolean isPriceValid(CryptoPrice price) {
        return price.getTimestamp().isAfter(LocalDateTime.now().minusMinutes(5));
    }
    
    /**
     * 获取价格变化百分比
     */
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