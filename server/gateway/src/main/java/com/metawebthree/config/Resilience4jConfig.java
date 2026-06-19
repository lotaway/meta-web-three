package com.metawebthree.Config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.timelimiter.TimeLimiterRegistry;
import io.github.resilience4j.retry.RetryRegistry;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import io.github.resilience4j.bulkhead.BulkheadRegistry;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/**
 * Resilience4j 配置类
 * 启用 Resilience4j 的 AOP 支持，并配置各种容错机制
 */
@Configuration
@EnableConfigurationProperties
public class Resilience4jConfig {

    /**
     * 配置 CircuitBreaker Registry
     * 可以在这里添加自定义的 CircuitBreaker 配置
     */
    @Bean
    public CircuitBreakerRegistry circuitBreakerRegistry() {
        return CircuitBreakerRegistry.ofDefaults();
    }

    /**
     * 配置 TimeLimiter Registry
     */
    @Bean
    public TimeLimiterRegistry timeLimiterRegistry() {
        return TimeLimiterRegistry.ofDefaults();
    }

    /**
     * 配置 Retry Registry
     */
    @Bean
    public RetryRegistry retryRegistry() {
        return RetryRegistry.ofDefaults();
    }

    /**
     * 配置 RateLimiter Registry
     */
    @Bean
    public RateLimiterRegistry rateLimiterRegistry() {
        return RateLimiterRegistry.ofDefaults();
    }

    /**
     * 配置 Bulkhead Registry
     */
    @Bean
    public BulkheadRegistry bulkheadRegistry() {
        return BulkheadRegistry.ofDefaults();
    }
}
