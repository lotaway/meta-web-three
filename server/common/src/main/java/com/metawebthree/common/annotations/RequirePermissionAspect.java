package com.metawebthree.common.annotations;

import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.constants.HeaderConstants;
import jakarta.servlet.http.HttpServletRequest;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.server.ResponseStatusException;

import java.lang.reflect.Method;

@Aspect
@Component
public class RequirePermissionAspect {

    @Autowired(required = false)
    private PermissionChecker permissionChecker;

    @Autowired(required = false)
    private TokenBlacklistService tokenBlacklistService;

    @Pointcut("@annotation(com.metawebthree.common.annotations.RequirePermission)")
    public void permissionPointcut() {
    }

    @Around("permissionPointcut()")
    public Object checkPermission(ProceedingJoinPoint joinPoint) throws Throwable {
        if (permissionChecker == null) {
            return joinPoint.proceed();
        }
        if (tokenBlacklistService != null) {
            String originalToken = getRequestHeader("X-Original-Token");
            if (originalToken != null && tokenBlacklistService.isBlacklisted(originalToken)) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Token has been revoked");
            }
        }
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        RequirePermission annotation = method.getAnnotation(RequirePermission.class);
        Long userId = getUserIdFromRequest();
        if (userId == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Missing X-User-Id header");
        }
        if (!permissionChecker.hasPermission(userId, annotation.value())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Permission denied: " + annotation.value());
        }
        return joinPoint.proceed();
    }

    private String getRequestHeader(String name) {
        ServletRequestAttributes attrs = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attrs == null) {
            return null;
        }
        return attrs.getRequest().getHeader(name);
    }

    private Long getUserIdFromRequest() {
        ServletRequestAttributes attrs = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attrs == null) {
            return null;
        }
        HttpServletRequest request = attrs.getRequest();
        String userIdStr = request.getHeader(HeaderConstants.USER_ID);
        if (userIdStr == null || userIdStr.isBlank()) {
            return null;
        }
        return Long.parseLong(userIdStr);
    }
}
