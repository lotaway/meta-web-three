package com.metawebthree.common.annotations;

import java.lang.reflect.Method;
import java.util.List;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.server.ResponseStatusException;

import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.constants.HeaderConstants;

import jakarta.servlet.http.HttpServletRequest;

/**
 * Enforces {@link RequirePermission} in servlet services that provide at least
 * one {@link PermissionChecker}.
 *
 * <p>When several checkers are present (for example the digital-twin and MES
 * checkers inside factory-service), permission is granted when any checker
 * grants it: each checker owns a disjoint set of permission codes. The aspect is
 * created by {@code PermissionAutoConfiguration}, not by component scanning, so
 * a service can no longer silently disable every permission annotation by
 * omitting one package from {@code scanBasePackages}.</p>
 */
@Aspect
public class RequirePermissionAspect {

    private final ObjectProvider<PermissionChecker> permissionCheckers;
    private final ObjectProvider<TokenBlacklistService> tokenBlacklistService;

    public RequirePermissionAspect(ObjectProvider<PermissionChecker> permissionCheckers,
                                   ObjectProvider<TokenBlacklistService> tokenBlacklistService) {
        this.permissionCheckers = permissionCheckers;
        this.tokenBlacklistService = tokenBlacklistService;
    }

    @Pointcut("@annotation(com.metawebthree.common.annotations.RequirePermission)")
    public void permissionPointcut() {
    }

    @Around("permissionPointcut()")
    public Object checkPermission(ProceedingJoinPoint joinPoint) throws Throwable {
        List<PermissionChecker> checkers = permissionCheckers.orderedStream().toList();
        if (checkers.isEmpty()) {
            return joinPoint.proceed();
        }
        TokenBlacklistService blacklist = tokenBlacklistService.getIfAvailable();
        if (blacklist != null) {
            String originalToken = getRequestHeader("X-Original-Token");
            if (originalToken != null && blacklist.isBlacklisted(originalToken)) {
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
        boolean granted = checkers.stream()
                .anyMatch(checker -> checker.hasPermission(userId, annotation.value()));
        if (!granted) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN,
                    "Permission denied: " + annotation.value());
        }
        return joinPoint.proceed();
    }

    private String getRequestHeader(String name) {
        ServletRequestAttributes attributes = currentRequestAttributes();
        return attributes == null ? null : attributes.getRequest().getHeader(name);
    }

    private Long getUserIdFromRequest() {
        ServletRequestAttributes attributes = currentRequestAttributes();
        if (attributes == null) {
            return null;
        }
        HttpServletRequest request = attributes.getRequest();
        String userId = request.getHeader(HeaderConstants.USER_ID);
        if (userId == null || userId.isBlank()) {
            return null;
        }
        try {
            return Long.parseLong(userId);
        } catch (NumberFormatException exception) {
            return null;
        }
    }

    private ServletRequestAttributes currentRequestAttributes() {
        return RequestContextHolder.getRequestAttributes() instanceof ServletRequestAttributes attributes
                ? attributes
                : null;
    }
}
