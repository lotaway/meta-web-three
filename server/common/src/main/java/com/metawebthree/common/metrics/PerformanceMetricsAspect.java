package com.metawebthree.common.metrics;

import com.metawebthree.common.alert.AlertService;
import io.micrometer.core.instrument.MeterRegistry;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * AOP aspect for collecting performance metrics from service methods
 */
@Aspect
@Component
public class PerformanceMetricsAspect {
    
    @Autowired
    private PerformanceMetricsService metricsService;
    
    @Autowired
    private AlertService alertService;
    
    /**
     * Monitor all service and Dubbo service methods
     */
    @Around("@within(org.springframework.stereotype.Service) || @within(org.apache.dubbo.config.annotation.DubboService)")
    public Object monitorPerformance(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        long startTime = System.currentTimeMillis();
        
        try {
            // Record request
            metricsService.recordRequest(className, methodName);
            
            // Execute method
            Object result = joinPoint.proceed();
            
            // Record success and execution time
            long executionTime = System.currentTimeMillis() - startTime;
            metricsService.recordExecutionTime(className, methodName, executionTime);
            metricsService.recordSuccess(className, methodName);
            
            // Alert if method execution is too slow (over 3 seconds)
            if (executionTime > 3000) {
                alertService.triggerAlert(
                    "SLOW_METHOD",
                    String.format("Method %s.%s executed in %d ms", className, methodName, executionTime)
                );
            }
            
            return result;
            
        } catch (Exception ex) {
            // Record error and execution time
            long executionTime = System.currentTimeMillis() - startTime;
            metricsService.recordExecutionTime(className, methodName, executionTime);
            metricsService.recordError(className, methodName, ex.getClass().getSimpleName());
            
            // Record exception for tracing
            alertService.recordException(className, methodName, ex);
            
            // Re-throw the exception
            throw ex;
        }
    }
}
