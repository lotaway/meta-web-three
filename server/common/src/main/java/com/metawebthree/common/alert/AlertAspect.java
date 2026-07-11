package com.metawebthree.common.alert;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.stereotype.Component;

/**
 * AOP aspect for monitoring method execution and triggering alerts on exceptions
 */
@Aspect
@Component
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
public class AlertAspect {
    
    @Autowired
    private AlertService alertService;
    
    /**
     * Monitor service layer methods for exceptions and performance issues
     */
    @Around("(@within(org.springframework.stereotype.Service) || @within(org.apache.dubbo.config.annotation.DubboService)) && !target(com.metawebthree.common.metrics.PerformanceMetricsService)")
    public Object monitorServiceMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        long startTime = System.currentTimeMillis();
        
        try {
            Object result = joinPoint.proceed();
            
            long executionTime = System.currentTimeMillis() - startTime;
            
            // Alert if method execution is too slow (over 5 seconds)
            if (executionTime > 5000) {
                alertService.triggerAlert(
                    "SLOW_METHOD",
                    String.format("Method %s.%s executed in %d ms", className, methodName, executionTime)
                );
            }
            
            return result;
            
        } catch (Exception ex) {
            long executionTime = System.currentTimeMillis() - startTime;
            
            // Record exception for tracing
            alertService.recordException(className, methodName, ex);
            
            // Re-throw the exception
            throw ex;
        }
    }
    
    /**
     * Monitor controller layer methods for exceptions
     */
    @Around("@within(org.springframework.web.bind.annotation.RestController)")
    public Object monitorControllerMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        try {
            return joinPoint.proceed();
        } catch (Exception ex) {
            // Record exception for tracing
            alertService.recordException(className, methodName, ex);
            throw ex;
        }
    }
}
