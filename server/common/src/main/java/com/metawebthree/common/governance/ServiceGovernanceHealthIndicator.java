package com.metawebthree.common.governance;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Health indicator for service governance components
 * 
 * Reports health status of circuit breakers and rate limiters
 * for monitoring and observability.
 */
@Slf4j
@RequiredArgsConstructor
public class ServiceGovernanceHealthIndicator implements HealthIndicator {

    private final CircuitBreakerRegistry circuitBreakerRegistry;
    private final RateLimiterRegistry rateLimiterRegistry;

    @Override
    public Health health() {
        Map<String, Object> details = new HashMap<>();
        
        // Check circuit breakers
        Map<String, String> circuitBreakerStatus = checkCircuitBreakers();
        details.put("circuitBreakers", circuitBreakerStatus);
        
        // Check rate limiters
        Map<String, String> rateLimiterStatus = checkRateLimiters();
        details.put("rateLimiters", rateLimiterStatus);
        
        // Determine overall status
        boolean hasOpenCircuitBreaker = circuitBreakerStatus.values().stream()
                .anyMatch(status -> "OPEN".equals(status));
        
        if (hasOpenCircuitBreaker) {
            return Health.status(new Status("DEGRADED", "One or more circuit breakers are open"))
                    .withDetails(details)
                    .build();
        }
        
        return Health.up()
                .withDetails(details)
                .build();
    }

    private Map<String, String> checkCircuitBreakers() {
        Map<String, String> status = new HashMap<>();
        
        circuitBreakerRegistry.getAllCircuitBreakers().forEach(cb -> {
            String name = cb.getName();
            CircuitBreaker.State state = cb.getState();
            status.put(name, state.name());
            
            // Log detailed metrics
            CircuitBreaker.Metrics metrics = cb.getMetrics();
            log.debug("CircuitBreaker '{}' metrics: failureRate={}, slowCallRate={}, bufferedCalls={}, failedCalls={}",
                    name,
                    metrics.getFailureRate(),
                    metrics.getSlowCallRate(),
                    metrics.getNumberOfBufferedCalls(),
                    metrics.getNumberOfFailedCalls());
        });
        
        return status;
    }

    private Map<String, String> checkRateLimiters() {
        Map<String, String> status = new HashMap<>();
        
        rateLimiterRegistry.getAllRateLimiters().forEach(rl -> {
            String name = rl.getName();
            RateLimiter.Metrics metrics = rl.getMetrics();
            
            // Report if rate limiter is under pressure
            int availablePermissions = metrics.getAvailablePermissions();
            int numberOfWaitingThreads = metrics.getNumberOfWaitingThreads();
            
            if (numberOfWaitingThreads > 10) {
                status.put(name, "PRESSURED");
                log.warn("RateLimiter '{}' is under pressure: availablePermissions={}, waitingThreads={}",
                        name, availablePermissions, numberOfWaitingThreads);
            } else {
                status.put(name, "NORMAL");
            }
        });
        
        return status;
    }

    /**
     * Get detailed metrics for a specific circuit breaker
     */
    public Map<String, Object> getCircuitBreakerMetrics(String name) {
        Optional<CircuitBreaker> cbOpt = circuitBreakerRegistry.getAllCircuitBreakers().stream()
                .filter(cb -> cb.getName().equals(name))
                .findFirst();
        
        if (cbOpt.isEmpty()) {
            return Map.of("error", "CircuitBreaker not found: " + name);
        }
        
        CircuitBreaker cb = cbOpt.get();
        CircuitBreaker.Metrics metrics = cb.getMetrics();
        return Map.of(
                "name", name,
                "state", cb.getState().name(),
                "failureRate", metrics.getFailureRate(),
                "slowCallRate", metrics.getSlowCallRate(),
                "bufferedCalls", metrics.getNumberOfBufferedCalls(),
                "failedCalls", metrics.getNumberOfFailedCalls(),
                "slowCalls", metrics.getNumberOfSlowCalls(),
                "successfulCalls", metrics.getNumberOfSuccessfulCalls()
        );
    }

    /**
     * Get detailed metrics for a specific rate limiter
     */
    public Map<String, Object> getRateLimiterMetrics(String name) {
        Optional<RateLimiter> rlOpt = rateLimiterRegistry.getAllRateLimiters().stream()
                .filter(rl -> rl.getName().equals(name))
                .findFirst();
        
        if (rlOpt.isEmpty()) {
            return Map.of("error", "RateLimiter not found: " + name);
        }
        
        RateLimiter rl = rlOpt.get();
        RateLimiter.Metrics metrics = rl.getMetrics();
        return Map.of(
                "name", name,
                "availablePermissions", metrics.getAvailablePermissions(),
                "numberOfWaitingThreads", metrics.getNumberOfWaitingThreads()
        );
    }
}
