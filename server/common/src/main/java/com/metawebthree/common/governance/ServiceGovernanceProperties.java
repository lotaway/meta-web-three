package com.metawebthree.common.governance;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.time.Duration;

/**
 * Service Governance Configuration Properties
 * 
 * Can be customized via application.yml:
 * <pre>
 * service-governance:
 *   circuit-breaker:
 *     enabled: true
 *     sliding-window-size: 10
 *     failure-rate-threshold: 50
 *   rate-limiter:
 *     enabled: true
 *     limit-for-period: 100
 *   retry:
 *     enabled: true
 *     max-attempts: 3
 * </pre>
 */
@Data
@Component
@ConfigurationProperties(prefix = "service-governance")
public class ServiceGovernanceProperties {

    private CircuitBreakerProperties circuitBreaker = new CircuitBreakerProperties();
    private RateLimiterProperties rateLimiter = new RateLimiterProperties();
    private RetryProperties retry = new RetryProperties();
    private boolean enabled = true;

    @Data
    public static class CircuitBreakerProperties {
        private boolean enabled = true;
        private int slidingWindowSize = 10;
        private int minimumNumberOfCalls = 5;
        private float failureRateThreshold = 50f;
        private float slowCallRateThreshold = 100f;
        private Duration slowCallDurationThreshold = Duration.ofSeconds(2);
        private Duration waitDurationInOpenState = Duration.ofSeconds(10);
        private int permittedNumberOfCallsInHalfOpenState = 3;
        private boolean automaticTransitionFromOpenToHalfOpenEnabled = true;
    }

    @Data
    public static class RateLimiterProperties {
        private boolean enabled = true;
        private int limitForPeriod = 100;
        private Duration limitRefreshPeriod = Duration.ofSeconds(1);
        private Duration timeoutDuration = Duration.ofSeconds(5);
    }

    @Data
    public static class RetryProperties {
        private boolean enabled = true;
        private int maxAttempts = 3;
        private Duration waitDuration = Duration.ofMillis(500);
        private boolean exponentialBackoffEnabled = true;
        private double exponentialBackoffMultiplier = 2.0;
    }
}
