package com.metawebthree.common.annotations;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.reflect.Method;
import java.util.Arrays;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.reflect.MethodSignature;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.server.ResponseStatusException;

import com.metawebthree.common.auth.TokenBlacklistService;

class RequirePermissionAspectTest {

    private static final String PERMISSION = "device:read";

    @RequirePermission(PERMISSION)
    public void guarded() {
    }

    @BeforeEach
    void setUp() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-User-Id", "7");
        RequestContextHolder.setRequestAttributes(
                new ServletRequestAttributes(request, new MockHttpServletResponse()));
    }

    @AfterEach
    void tearDown() {
        RequestContextHolder.resetRequestAttributes();
    }

    @Test
    void proceedsWhenNoCheckerIsRegistered() throws Throwable {
        RequirePermissionAspect aspect = aspect(provider(), blacklistProvider(null));
        assertEquals("ok", aspect.checkPermission(joinPoint()));
    }

    @Test
    void rejectsMissingUserIdHeader() throws Throwable {
        MockHttpServletRequest request = new MockHttpServletRequest();
        RequestContextHolder.setRequestAttributes(
                new ServletRequestAttributes(request, new MockHttpServletResponse()));

        RequirePermissionAspect aspect = aspect(provider((id, permission) -> true), blacklistProvider(null));
        ResponseStatusException exception = assertThrows(
                ResponseStatusException.class, () -> aspect.checkPermission(joinPoint()));
        assertEquals(HttpStatus.UNAUTHORIZED, exception.getStatusCode());
    }

    @Test
    void grantsWhenAnyCheckerAllowsThePermission() throws Throwable {
        RequirePermissionAspect aspect = aspect(
                provider((id, permission) -> false, (id, permission) -> true),
                blacklistProvider(null));
        assertEquals("ok", aspect.checkPermission(joinPoint()));
    }

    @Test
    void rejectsWhenNoCheckerAllowsThePermission() throws Throwable {
        RequirePermissionAspect aspect = aspect(
                provider((id, permission) -> false), blacklistProvider(null));
        ResponseStatusException exception = assertThrows(
                ResponseStatusException.class, () -> aspect.checkPermission(joinPoint()));
        assertEquals(HttpStatus.FORBIDDEN, exception.getStatusCode());
    }

    @Test
    void rejectsBlacklistedOriginalToken() throws Throwable {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Original-Token", "Bearer revoked");
        RequestContextHolder.setRequestAttributes(
                new ServletRequestAttributes(request, new MockHttpServletResponse()));

        TokenBlacklistService blacklist = Mockito.mock(TokenBlacklistService.class);
        Mockito.when(blacklist.isBlacklisted("Bearer revoked")).thenReturn(true);
        RequirePermissionAspect aspect = aspect(provider((id, permission) -> true), blacklistProvider(blacklist));

        ResponseStatusException exception = assertThrows(
                ResponseStatusException.class, () -> aspect.checkPermission(joinPoint()));
        assertEquals(HttpStatus.UNAUTHORIZED, exception.getStatusCode());
    }

    @SuppressWarnings("unchecked")
    private ObjectProvider<PermissionChecker> provider(PermissionChecker... checkers) {
        ObjectProvider<PermissionChecker> provider = Mockito.mock(ObjectProvider.class);
        Mockito.when(provider.orderedStream())
                .thenAnswer(invocation -> Arrays.stream(checkers));
        return provider;
    }

    @SuppressWarnings("unchecked")
    private ObjectProvider<TokenBlacklistService> blacklistProvider(TokenBlacklistService service) {
        ObjectProvider<TokenBlacklistService> provider = Mockito.mock(ObjectProvider.class);
        Mockito.when(provider.getIfAvailable()).thenReturn(service);
        return provider;
    }

    private ProceedingJoinPoint joinPoint() throws Throwable {
        Method method = RequirePermissionAspectTest.class.getMethod("guarded");
        MethodSignature signature = Mockito.mock(MethodSignature.class);
        Mockito.when(signature.getMethod()).thenReturn(method);
        ProceedingJoinPoint joinPoint = Mockito.mock(ProceedingJoinPoint.class);
        Mockito.when(joinPoint.getSignature()).thenReturn(signature);
        Mockito.when(joinPoint.proceed()).thenReturn("ok");
        return joinPoint;
    }

    private RequirePermissionAspect aspect(ObjectProvider<PermissionChecker> checkers,
                                           ObjectProvider<TokenBlacklistService> blacklist) {
        return new RequirePermissionAspect(checkers, blacklist);
    }
}
