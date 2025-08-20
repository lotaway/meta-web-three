package com.metawebthree.common.annotations;

import java.lang.reflect.Method;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.aspectj.lang.reflect.MethodSignature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class LogMethodAspect {

    private final Logger log = LoggerFactory.getLogger(this.getClass());

    @Pointcut("@annotation(com.metawebthree.common.annotations.LogMethod)")
    public void logMethodPointcut() {
    }

    @Before("logMethodPointcut()")
    public void logBefore(JoinPoint joinPoint) {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        LogMethod logMethod = method.getAnnotation(LogMethod.class);
        if (!logMethod.param()) {
            return;
        }
        log.info("Entering method: {} with parameters: {}",
                joinPoint.getSignature().getName(),
                joinPoint.getArgs());
    }

    @AfterReturning(pointcut = "logMethodPointcut()", returning = "result")
    public void logAfterReturning(JoinPoint joinPoint, Object result) {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        LogMethod logMethod = method.getAnnotation(LogMethod.class);
        if (!logMethod.result()) {
            return;
        }
        log.info("Exiting method: {} with result: {}",
                joinPoint.getSignature().getName(),
                result);
    }

    @Around("logMethodPointcut()")
    public Object logAround(ProceedingJoinPoint joinPoint) throws Throwable {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        LogMethod logMethod = method.getAnnotation(LogMethod.class);
        if (!logMethod.timeCost()) {
            return joinPoint.proceed();
        }
        long startTime = System.currentTimeMillis();
        Object result = joinPoint.proceed();
        long executionTime = System.currentTimeMillis() - startTime;
        log.info("Method: {} executed in: {} ms",
                joinPoint.getSignature().getName(),
                executionTime);
        return result;
    }
}
