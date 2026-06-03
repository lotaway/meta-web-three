package com.metawebthree.common.governance;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

/**
 * Service Governance Aspect
 * 
 * Automatically applies circuit breaker and rate limiter to Dubbo RPC calls
 * based on the service interface name.
 * 
 * Order:
 * 1. Rate Limiter (first line of defense)
 * 2. Circuit Breaker (second line of defense)
 */
@Aspect
@Component
@Order(1)
@Slf4j
@RequiredArgsConstructor
public class ServiceGovernanceAspect {

    private final CircuitBreakerRegistry circuitBreakerRegistry;
    private final RateLimiterRegistry rateLimiterRegistry;
    private final ServiceGovernanceProperties properties;

    /**
     * Apply governance to all @DubboService method calls
     */
    @Around("@within(org.apache.dubbo.config.annotation.DubboService) || " +
            "@annotation(org.apache.dubbo.config.annotation.DubboService)")
    public Object governDubboServiceCall(ProceedingJoinPoint joinPoint) throws Throwable {
        if (!properties.isEnabled()) {
            return joinPoint.proceed();
        }

        String serviceName = getServiceName(joinPoint);
        String methodName = getMethodName(joinPoint);
        String circuitBreakerName = serviceName;
        String rateLimiterName = serviceName + "-" + methodName;

        log.debug("Applying governance to {}.{}", serviceName, methodName);

        try {
            // Apply rate limiter first
            if (properties.getRateLimiter().isEnabled()) {
                RateLimiter rateLimiter = getOrCreateRateLimiter(rateLimiterName);
                return RateLimiter.decorateSupplier(rateLimiter, () -> {
                    try {
                        return applyCircuitBreaker(joinPoint, circuitBreakerName);
                    } catch (Throwable e) {
                        throw new RuntimeException(e);
                    }
                }).get();
            } else {
                return applyCircuitBreaker(joinPoint, circuitBreakerName);
            }
        } catch (RuntimeException e) {
            if (e.getCause() != null) {
                throw e.getCause();
            }
            throw e;
        }
    }

    private Object applyCircuitBreaker(ProceedingJoinPoint joinPoint, String circuitBreakerName) throws Throwable {
        if (!properties.getCircuitBreaker().isEnabled()) {
            return joinPoint.proceed();
        }

        CircuitBreaker circuitBreaker = getOrCreateCircuitBreaker(circuitBreakerName);
        return CircuitBreaker.decorateSupplier(circuitBreaker, () -> {
            try {
                return joinPoint.proceed();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }).get();
    }

    private CircuitBreaker getOrCreateCircuitBreaker(String name) {
        return circuitBreakerRegistry.getAllCircuitBreakers().stream()
                .filter(cb -> cb.getName().equals(name))
                .findFirst()
                .orElseGet(() -> {
                    CircuitBreakerConfig config = CircuitBreakerConfig.custom()
                            .slidingWindowSize(properties.getCircuitBreaker().getSlidingWindowSize())
                            .minimumNumberOfCalls(properties.getCircuitBreaker().getMinimumNumberOfCalls())
                            .failureRateThreshold(properties.getCircuitBreaker().getFailureRateThreshold())
                            .slowCallRateThreshold(properties.getCircuitBreaker().getSlowCallRateThreshold())
                            .slowCallDurationThreshold(properties.getCircuitBreaker().getSlowCallDurationThreshold())
                            .waitDurationInOpenState(properties.getCircuitBreaker().getWaitDurationInOpenState())
                            .permittedNumberOfCallsInHalfOpenState(properties.getCircuitBreaker().getPermittedNumberOfCallsInHalfOpenState())
                            .automaticTransitionFromOpenToHalfOpenEnabled(properties.getCircuitBreaker().isAutomaticTransitionFromOpenToHalfOpenEnabled())
                            .build();
                    CircuitBreaker cb = circuitBreakerRegistry.circuitBreaker(name, config);
                    log.info("Created CircuitBreaker: {}", name);
                    return cb;
                });
    }

    private RateLimiter getOrCreateRateLimiter(String name) {
        return rateLimiterRegistry.getAllRateLimiters().stream()
                .filter(rl -> rl.getName().equals(name))
                .findFirst()
                .orElseGet(() -> {
                    RateLimiterConfig config = RateLimiterConfig.custom()
                            .limitForPeriod(properties.getRateLimiter().getLimitForPeriod())
                            .limitRefreshPeriod(properties.getRateLimiter().getLimitRefreshPeriod())
                            .timeoutDuration(properties.getRateLimiter().getTimeoutDuration())
                            .build();
                    RateLimiter rl = rateLimiterRegistry.rateLimiter(name, config);
                    log.info("Created RateLimiter: {}", name);
                    return rl;
                });
    }

    private String getServiceName(ProceedingJoinPoint joinPoint) {
        Class<?> targetClass = joinPoint.getTarget().getClass();
        
        // Try to get the interface name
        Class<?>[] interfaces = targetClass.getInterfaces();
        if (interfaces.length > 0) {
            String interfaceName = interfaces[0].getSimpleName();
            // Remove "Service" suffix if present for cleaner naming
            if (interfaceName.endsWith("Service")) {
                return interfaceName.substring(0, interfaceName.length() - 7);
            }
            return interfaceName;
        }
        
        return targetClass.getSimpleName();
    }

    private String getMethodName(ProceedingJoinPoint joinPoint) {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        return signature.getMethod().getName();
    }

    /**
     * Get circuit breaker state for monitoring
     */
    public CircuitBreaker.State getCircuitBreakerState(String serviceName) {
        return circuitBreakerRegistry.getAllCircuitBreakers().stream()
                .filter(cb -> cb.getName().equals(serviceName))
                .map(CircuitBreaker::getState)
                .findFirst()
                .orElse(null);
    }

    /**
     * Force open circuit breaker (for testing/maintenance)
     */
    public void forceOpenCircuitBreaker(String serviceName) {
        circuitBreakerRegistry.getAllCircuitBreakers().stream()
                .filter(cb -> cb.getName().equals(serviceName))
                .findFirst()
                .ifPresent(cb -> {
                    cb.transitionToOpenState();
                    log.warn("CircuitBreaker {} forced to OPEN state", serviceName);
                });
    }

    /**
     * Close circuit breaker (for recovery)
     */
    public void closeCircuitBreaker(String serviceName) {
        circuitBreakerRegistry.getAllCircuitBreakers().stream()
                .filter(cb -> cb.getName().equals(serviceName))
                .findFirst()
                .ifPresent(cb -> {
                    cb.transitionToClosedState();
                    log.info("CircuitBreaker {} forced to CLOSED state", serviceName);
                });
    }
}
