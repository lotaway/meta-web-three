package com.metawebthree.payment.application;

import com.metawebthree.payment.config.PriceEngineProperties;
import com.metawebthree.payment.domain.exception.ExternalServiceException;
import com.metawebthree.payment.domain.model.CryptoPrice;
import com.metawebthree.payment.infrastructure.persistence.mapper.CryptoPriceRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Service
@RequiredArgsConstructor
@Slf4j
public class PriceEngineServiceImpl {

    private static final String[] SYMBOLS = { "BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD" };

    private final CryptoPriceRepository cryptoPriceRepository;
    private final ExternalPriceServiceImpl externalPriceService;
    private final PriceEngineProperties properties;

    // In-memory cache refreshed by the scheduled task; price data is too volatile for
    // Spring Cache and cheap enough to re-read from the database on a miss.
    private final Map<String, CryptoPrice> priceCache = new ConcurrentHashMap<>();
    private final AtomicInteger consecutiveFailures = new AtomicInteger(0);
    private volatile long circuitOpenUntil = 0L;

    public CryptoPrice getCurrentPrice(String symbol) {
        CryptoPrice cachedPrice = priceCache.get(symbol);
        if (cachedPrice != null && isPriceValid(cachedPrice)) {
            return cachedPrice;
        }
        CryptoPrice dbPrice = cryptoPriceRepository.findFirstBySymbolOrderByTimestampDesc(symbol);
        if (dbPrice != null) {
            return dbPrice;
        }
        return fetchWithFallback(symbol);
    }

    public BigDecimal getWeightedAveragePrice(String baseCurrency, String quoteCurrency) {
        List<CryptoPrice> prices = cryptoPriceRepository.findByBaseCurrencyAndQuoteCurrency(baseCurrency, quoteCurrency);
        if (prices.isEmpty()) {
            throw new ExternalServiceException("PriceEngine",
                    "No price data available for " + baseCurrency + "-" + quoteCurrency);
        }
        BigDecimal totalWeight = BigDecimal.ZERO;
        BigDecimal weightedSum = BigDecimal.ZERO;
        Map<String, BigDecimal> weights = Map.of(
                "binance", new BigDecimal("0.4"),
                "coinbase", new BigDecimal("0.3"),
                "okx", new BigDecimal("0.3"));
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

    /**
     * Periodic refresh. Uses a circuit breaker so that when every external source is
     * unreachable the task backs off for a cooldown window instead of hammering the
     * (slow) exchanges and flooding the logs every cycle.
     */
    @Scheduled(fixedRate = 5000)
    public void updatePrices() {
        if (!properties.isEnabled() || isCircuitOpen()) {
            return;
        }
        int failures = 0;
        for (String symbol : SYMBOLS) {
            failures += refresh(symbol) ? 0 : 1;
        }
        recordCycleResult(failures);
    }

    /** Best-effort single-symbol refresh used by the scheduled task; never throws. */
    private boolean refresh(String symbol) {
        try {
            CryptoPrice external = externalPriceService.fetchPrice(symbol);
            if (external != null) {
                persistAndCache(symbol, external);
                return true;
            }
            return false;
        } catch (Exception e) {
            // Expected when offline; the circuit breaker handles the aggregate outcome.
            log.debug("Failed to update price for {}: {}", symbol, e.getMessage());
            return false;
        }
    }

    /** Full resolution chain used by request paths; throws only when nothing is available. */
    private CryptoPrice fetchWithFallback(String symbol) {
        try {
            CryptoPrice external = externalPriceService.fetchPrice(symbol);
            if (external != null) {
                return persistAndCache(symbol, external);
            }
        } catch (Exception e) {
            log.debug("Failed to fetch external price for {}: {}", symbol, e.getMessage());
        }
        CryptoPrice cached = priceCache.get(symbol);
        if (cached != null) {
            return cached;
        }
        return mockOrThrow(symbol);
    }

    private CryptoPrice persistAndCache(String symbol, CryptoPrice price) {
        cryptoPriceRepository.insert(price);
        priceCache.put(symbol, price);
        return price;
    }

    private CryptoPrice mockOrThrow(String symbol) {
        BigDecimal mock = properties.getMockPrices().get(symbol);
        if (mock == null) {
            throw new ExternalServiceException("PriceEngine", "Unable to fetch price for " + symbol);
        }
        return toMockPrice(symbol, mock);
    }

    private CryptoPrice toMockPrice(String symbol, BigDecimal price) {
        String[] parts = symbol.split("-");
        return CryptoPrice.builder()
                .symbol(symbol)
                .baseCurrency(parts[0])
                .quoteCurrency(parts[1])
                .price(price)
                .source("mock")
                .timestamp(LocalDateTime.now())
                .build();
    }

    private boolean isCircuitOpen() {
        return System.currentTimeMillis() < circuitOpenUntil;
    }

    private void recordCycleResult(int failures) {
        if (failures < SYMBOLS.length) {
            consecutiveFailures.set(0);
            return;
        }
        int count = consecutiveFailures.incrementAndGet();
        if (count >= properties.getMaxConsecutiveFailures()) {
            openCircuit(count);
        }
    }

    private void openCircuit(int count) {
        long now = System.currentTimeMillis();
        if (now < circuitOpenUntil) {
            return;
        }
        long cooldownMs = properties.getFailureCooldownSeconds() * 1000L;
        circuitOpenUntil = now + cooldownMs;
        consecutiveFailures.set(0);
        log.warn("All price sources unreachable; pausing external price refresh for {}s",
                properties.getFailureCooldownSeconds());
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
        return change.divide(oldest.getPrice(), 4, RoundingMode.HALF_UP).multiply(new BigDecimal("100"));
    }
}
