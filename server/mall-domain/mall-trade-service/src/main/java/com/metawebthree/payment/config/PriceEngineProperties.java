package com.metawebthree.payment.config;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import lombok.Data;

/**
 * Configuration for the crypto price engine.
 *
 * <p>Environment-specific behaviour (offline / air-gapped deployments) is driven from
 * {@code payment.price-engine.*}: external refresh can be disabled, retried-then-paused
 * via a circuit breaker, and seeded with static {@code mock-prices} so downstream code
 * never fails purely because a remote exchange is unreachable.</p>
 */
@Data
@Component
@ConfigurationProperties(prefix = "payment.price-engine")
public class PriceEngineProperties {

    /** When false the scheduled external refresh is skipped entirely. */
    private boolean enabled = true;

    /** Consecutive all-sources-failed cycles before the circuit opens. */
    private int maxConsecutiveFailures = 3;

    /** How long (seconds) external refresh is paused once the circuit opens. */
    private int failureCooldownSeconds = 60;

    /**
     * Static per-symbol fallback prices (e.g. {@code BTC-USD: 50000}) used only when the
     * external sources are unreachable and there is no cached or persisted price. Left
     * empty by default so production always prefers live market data.
     */
    private Map<String, BigDecimal> mockPrices = new HashMap<>();
}
