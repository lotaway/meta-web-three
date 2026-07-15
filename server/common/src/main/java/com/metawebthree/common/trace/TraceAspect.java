package com.metawebthree.common.trace;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.stereotype.Component;

/**
 * AOP aspect for distributed tracing of service method calls
 * Automatically creates and manages trace spans
 */
@Aspect
@Component
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
public class TraceAspect {
    
    @Autowired
    private DistributedTraceService traceService;
    
    /**
     * Trace all service and Dubbo service method calls
     */
    @Around("(@within(org.springframework.stereotype.Service) || @within(org.apache.dubbo.config.annotation.DubboService)) && !within(com.metawebthree.common..*)")
    public Object traceMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        // Start trace span
        TraceSpan span = traceService.startSpan(className, methodName);
        
        try {
            // Execute the method
            Object result = joinPoint.proceed();
            
            // End span successfully
            traceService.endSpan(span, "SUCCESS", null);
            
            return result;
            
        } catch (Exception ex) {
            // End span with error
            traceService.endSpan(span, "ERROR", ex.getMessage());
            
            // Re-throw the exception
            throw ex;
        }
    }
    
    /**
     * Trace controller methods (entry points)
     */
    @Around("@within(org.springframework.web.bind.annotation.RestController) && !within(com.metawebthree.common..*)")
    public Object traceController(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        // Start trace span for controller
        TraceSpan span = traceService.startSpan(className, methodName);
        
        try {
            Object result = joinPoint.proceed();
            traceService.endSpan(span, "SUCCESS", null);
            return result;
            
        } catch (Exception ex) {
            traceService.endSpan(span, "ERROR", ex.getMessage());
            throw ex;
        }
    }
}
