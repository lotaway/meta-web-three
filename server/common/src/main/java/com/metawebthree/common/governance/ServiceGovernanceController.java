package com.metawebthree.common.governance;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Service Governance Controller
 * 
 * Provides REST endpoints for monitoring and managing circuit breakers and rate limiters.
 * 
 * Endpoints:
 * - GET /governance/health - Overall health status
 * - GET /governance/circuit-breakers - List all circuit breakers
 * - GET /governance/circuit-breakers/{name} - Get specific circuit breaker details
 * - POST /governance/circuit-breakers/{name}/open - Force open a circuit breaker
 * - POST /governance/circuit-breakers/{name}/close - Close a circuit breaker
 * - GET /governance/rate-limiters - List all rate limiters
 * - GET /governance/rate-limiters/{name} - Get specific rate limiter details
 */
@RestController
@RequestMapping("/governance")
@RequiredArgsConstructor
public class ServiceGovernanceController {

    private final CircuitBreakerRegistry circuitBreakerRegistry;
    private final RateLimiterRegistry rateLimiterRegistry;
    private final ServiceGovernanceHealthIndicator healthIndicator;

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> response = new HashMap<>();
        
        // Circuit breaker summary
        List<Map<String, Object>> cbList = StreamSupport.stream(
                circuitBreakerRegistry.getAllCircuitBreakers().spliterator(), false)
                .map(this::mapCircuitBreaker)
                .collect(Collectors.toList());
        
        long openCount = cbList.stream()
                .filter(cb -> "OPEN".equals(cb.get("state")))
                .count();
        
        // Rate limiter summary
        List<Map<String, Object>> rlList = StreamSupport.stream(
                rateLimiterRegistry.getAllRateLimiters().spliterator(), false)
                .map(this::mapRateLimiter)
                .collect(Collectors.toList());
        
        long pressuredCount = rlList.stream()
                .filter(rl -> "PRESSURED".equals(rl.get("status")))
                .count();
        
        response.put("status", openCount > 0 ? "DEGRADED" : "UP");
        response.put("circuitBreakers", Map.of(
                "total", cbList.size(),
                "open", openCount,
                "instances", cbList
        ));
        response.put("rateLimiters", Map.of(
                "total", rlList.size(),
                "pressured", pressuredCount,
                "instances", rlList
        ));
        
        return ResponseEntity.ok(response);
    }

    @GetMapping("/circuit-breakers")
    public ResponseEntity<List<Map<String, Object>>> listCircuitBreakers() {
        List<Map<String, Object>> list = StreamSupport.stream(
                circuitBreakerRegistry.getAllCircuitBreakers().spliterator(), false)
                .map(this::mapCircuitBreaker)
                .collect(Collectors.toList());
        return ResponseEntity.ok(list);
    }

    @GetMapping("/circuit-breakers/{name}")
    public ResponseEntity<Map<String, Object>> getCircuitBreaker(@PathVariable String name) {
        Optional<CircuitBreaker> cbOpt = StreamSupport.stream(
                circuitBreakerRegistry.getAllCircuitBreakers().spliterator(), false)
                .filter(cb -> cb.getName().equals(name))
                .findFirst();
        
        if (cbOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(mapCircuitBreakerDetails(cbOpt.get()));
    }

    @PostMapping("/circuit-breakers/{name}/open")
    public ResponseEntity<Map<String, Object>> openCircuitBreaker(@PathVariable String name) {
        Optional<CircuitBreaker> cbOpt = StreamSupport.stream(
                circuitBreakerRegistry.getAllCircuitBreakers().spliterator(), false)
                .filter(cb -> cb.getName().equals(name))
                .findFirst();
        
        if (cbOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        
        CircuitBreaker cb = cbOpt.get();
        cb.transitionToOpenState();
        return ResponseEntity.ok(Map.of(
                "name", name,
                "state", cb.getState().name(),
                "message", "Circuit breaker forced to OPEN state"
        ));
    }

    @PostMapping("/circuit-breakers/{name}/close")
    public ResponseEntity<Map<String, Object>> closeCircuitBreaker(@PathVariable String name) {
        Optional<CircuitBreaker> cbOpt = StreamSupport.stream(
                circuitBreakerRegistry.getAllCircuitBreakers().spliterator(), false)
                .filter(cb -> cb.getName().equals(name))
                .findFirst();
        
        if (cbOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        
        CircuitBreaker cb = cbOpt.get();
        cb.transitionToClosedState();
        return ResponseEntity.ok(Map.of(
                "name", name,
                "state", cb.getState().name(),
                "message", "Circuit breaker closed"
        ));
    }

    @GetMapping("/rate-limiters")
    public ResponseEntity<List<Map<String, Object>>> listRateLimiters() {
        List<Map<String, Object>> list = StreamSupport.stream(
                rateLimiterRegistry.getAllRateLimiters().spliterator(), false)
                .map(this::mapRateLimiter)
                .collect(Collectors.toList());
        return ResponseEntity.ok(list);
    }

    @GetMapping("/rate-limiters/{name}")
    public ResponseEntity<Map<String, Object>> getRateLimiter(@PathVariable String name) {
        Optional<RateLimiter> rlOpt = StreamSupport.stream(
                rateLimiterRegistry.getAllRateLimiters().spliterator(), false)
                .filter(rl -> rl.getName().equals(name))
                .findFirst();
        
        if (rlOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(mapRateLimiterDetails(rlOpt.get()));
    }

    private Map<String, Object> mapCircuitBreaker(CircuitBreaker cb) {
        CircuitBreaker.Metrics metrics = cb.getMetrics();
        return Map.of(
                "name", cb.getName(),
                "state", cb.getState().name(),
                "failureRate", metrics.getFailureRate(),
                "slowCallRate", metrics.getSlowCallRate(),
                "bufferedCalls", metrics.getNumberOfBufferedCalls(),
                "failedCalls", metrics.getNumberOfFailedCalls()
        );
    }

    private Map<String, Object> mapCircuitBreakerDetails(CircuitBreaker cb) {
        CircuitBreaker.Metrics metrics = cb.getMetrics();
        CircuitBreakerConfig config = cb.getCircuitBreakerConfig();
        Map<String, Object> details = new HashMap<>();
        details.put("name", cb.getName());
        details.put("state", cb.getState().name());
        details.put("metrics", Map.of(
                "failureRate", metrics.getFailureRate(),
                "slowCallRate", metrics.getSlowCallRate(),
                "bufferedCalls", metrics.getNumberOfBufferedCalls(),
                "failedCalls", metrics.getNumberOfFailedCalls(),
                "slowCalls", metrics.getNumberOfSlowCalls(),
                "successfulCalls", metrics.getNumberOfSuccessfulCalls(),
                "notPermittedCalls", metrics.getNumberOfNotPermittedCalls()
        ));
        details.put("config", Map.of(
                "slidingWindowSize", config.getSlidingWindowSize(),
                "minimumNumberOfCalls", config.getMinimumNumberOfCalls(),
                "failureRateThreshold", config.getFailureRateThreshold(),
                "slowCallDurationThreshold", config.getSlowCallDurationThreshold().toMillis() + "ms"
        ));
        return details;
    }

    private Map<String, Object> mapRateLimiter(RateLimiter rl) {
        RateLimiter.Metrics metrics = rl.getMetrics();
        String status = metrics.getNumberOfWaitingThreads() > 10 ? "PRESSURED" : "NORMAL";
        return Map.of(
                "name", rl.getName(),
                "status", status,
                "availablePermissions", metrics.getAvailablePermissions(),
                "waitingThreads", metrics.getNumberOfWaitingThreads()
        );
    }

    private Map<String, Object> mapRateLimiterDetails(RateLimiter rl) {
        RateLimiter.Metrics metrics = rl.getMetrics();
        Map<String, Object> details = new HashMap<>();
        details.put("name", rl.getName());
        details.put("metrics", Map.of(
                "availablePermissions", metrics.getAvailablePermissions(),
                "numberOfWaitingThreads", metrics.getNumberOfWaitingThreads()
        ));
        details.put("config", Map.of(
                "limitForPeriod", rl.getRateLimiterConfig().getLimitForPeriod(),
                "limitRefreshPeriod", rl.getRateLimiterConfig().getLimitRefreshPeriod().toMillis() + "ms",
                "timeoutDuration", rl.getRateLimiterConfig().getTimeoutDuration().toMillis() + "ms"
        ));
        return details;
    }
}
