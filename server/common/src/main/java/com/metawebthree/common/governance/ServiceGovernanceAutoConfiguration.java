package com.metawebthree.common.governance;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * Service Governance Auto Configuration
 * 
 * Provides unified configuration for:
 * - Circuit Breaker (熔断器)
 * - Rate Limiter (限流器)
 * - Retry (重试机制)
 * 
 * This configuration is automatically applied to all services that depend on the common module.
 */
@Configuration
@Slf4j
public class ServiceGovernanceAutoConfiguration {

    /**
     * Default Circuit Breaker configuration for backend services
     */
    @Bean
    @ConditionalOnMissingBean
    public CircuitBreakerConfig defaultCircuitBreakerConfig() {
        return CircuitBreakerConfig.custom()
                .slidingWindowType(CircuitBreakerConfig.SlidingWindowType.COUNT_BASED)
                .slidingWindowSize(10)
                .minimumNumberOfCalls(5)
                .failureRateThreshold(50)
                .slowCallDurationThreshold(Duration.ofSeconds(2))
                .slowCallRateThreshold(100)
                .permittedNumberOfCallsInHalfOpenState(3)
                .waitDurationInOpenState(Duration.ofSeconds(10))
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .recordExceptions(Exception.class)
                .ignoreExceptions(IllegalArgumentException.class, IllegalStateException.class)
                .build();
    }

    /**
     * Circuit Breaker Registry with default configuration
     */
    @Bean
    @ConditionalOnMissingBean
    public CircuitBreakerRegistry circuitBreakerRegistry(CircuitBreakerConfig defaultCircuitBreakerConfig) {
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.ofDefaults();
        registry.addConfiguration("default", defaultCircuitBreakerConfig);
        log.info("Applied default CircuitBreaker configuration");
        return registry;
    }

    /**
     * Default Rate Limiter configuration
     * Allows 100 requests per 1 second by default
     */
    @Bean
    @ConditionalOnMissingBean
    public RateLimiterConfig defaultRateLimiterConfig() {
        return RateLimiterConfig.custom()
                .limitForPeriod(100)
                .limitRefreshPeriod(Duration.ofSeconds(1))
                .timeoutDuration(Duration.ofSeconds(5))
                .build();
    }

    /**
     * Rate Limiter Registry with default configuration
     */
    @Bean
    @ConditionalOnMissingBean
    public RateLimiterRegistry rateLimiterRegistry(RateLimiterConfig defaultRateLimiterConfig) {
        RateLimiterRegistry registry = RateLimiterRegistry.ofDefaults();
        registry.addConfiguration("default", defaultRateLimiterConfig);
        log.info("Applied default RateLimiter configuration");
        return registry;
    }

    /**
     * Default Retry configuration
     * Retries up to 3 times with fixed backoff
     */
    @Bean
    @ConditionalOnMissingBean
    public RetryConfig defaultRetryConfig() {
        return RetryConfig.custom()
                .maxAttempts(3)
                .waitDuration(Duration.ofMillis(500))
                .retryExceptions(Exception.class)
                .ignoreExceptions(IllegalArgumentException.class, IllegalStateException.class)
                .build();
    }

    /**
     * Retry Registry with default configuration
     */
    @Bean
    @ConditionalOnMissingBean
    public RetryRegistry retryRegistry(RetryConfig defaultRetryConfig) {
        RetryRegistry registry = RetryRegistry.ofDefaults();
        registry.addConfiguration("default", defaultRetryConfig);
        log.info("Applied default Retry configuration");
        return registry;
    }

    /**
     * Service Governance Health Indicator
     * Provides health check for circuit breakers and rate limiters
     */
    @Bean
    @ConditionalOnMissingBean
    public ServiceGovernanceHealthIndicator serviceGovernanceHealthIndicator(
            CircuitBreakerRegistry circuitBreakerRegistry,
            RateLimiterRegistry rateLimiterRegistry) {
        return new ServiceGovernanceHealthIndicator(circuitBreakerRegistry, rateLimiterRegistry);
    }
}
