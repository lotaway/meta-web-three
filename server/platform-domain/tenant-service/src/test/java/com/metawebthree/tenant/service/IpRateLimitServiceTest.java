package com.metawebthree.tenant.service;

import jakarta.servlet.http.HttpServletRequest;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class IpRateLimitServiceTest {

    private static final int MAX_REQUESTS = 3;
    private static final Duration REFRESH_PERIOD = Duration.ofSeconds(60);

    @Mock
    private RedisTemplate<String, Object> redisTemplate;

    @Mock
    private ValueOperations<String, Object> valueOperations;

    @Mock
    private HttpServletRequest request;

    private IpRateLimitService rateLimitService;

    @BeforeEach
    void setUp() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        rateLimitService = new IpRateLimitService(redisTemplate);
        ReflectionTestUtils.setField(rateLimitService, "maxRequests", MAX_REQUESTS);
        ReflectionTestUtils.setField(rateLimitService, "refreshPeriod", REFRESH_PERIOD);
    }

    @Test
    void testIsAllowed_FirstRequest_SetsExpiry() {
        // Given
        when(request.getRemoteAddr()).thenReturn("192.168.1.10");
        when(valueOperations.increment(eq("metaweb:ratelimit:tenant:192.168.1.10"))).thenReturn(1L);

        // When & Then
        assertTrue(rateLimitService.isAllowed(request));
        verify(redisTemplate).expire("metaweb:ratelimit:tenant:192.168.1.10", REFRESH_PERIOD);
    }

    @Test
    void testIsAllowed_WithinLimit() {
        // Given
        when(request.getRemoteAddr()).thenReturn("192.168.1.10");
        when(valueOperations.increment(eq("metaweb:ratelimit:tenant:192.168.1.10"))).thenReturn(3L);

        // When & Then
        assertTrue(rateLimitService.isAllowed(request));
        verify(redisTemplate, never()).expire(anyKey(), anyDuration());
    }

    @Test
    void testIsBlocked_OverLimit() {
        // Given
        when(request.getRemoteAddr()).thenReturn("192.168.1.10");
        when(valueOperations.increment(eq("metaweb:ratelimit:tenant:192.168.1.10"))).thenReturn(4L);

        // When & Then
        assertFalse(rateLimitService.isAllowed(request));
        verify(redisTemplate, never()).expire(anyKey(), anyDuration());
    }

    @Test
    void testResolveClientIp_UsesForwardedForFirst() {
        // Given
        when(request.getHeader("X-Forwarded-For")).thenReturn("1.2.3.4, 10.0.0.1");
        when(valueOperations.increment(eq("metaweb:ratelimit:tenant:1.2.3.4"))).thenReturn(1L);

        // When
        rateLimitService.isAllowed(request);

        // Then
        verify(valueOperations).increment("metaweb:ratelimit:tenant:1.2.3.4");
    }

    private static String anyKey() {
        return org.mockito.ArgumentMatchers.anyString();
    }

    private static Duration anyDuration() {
        return org.mockito.ArgumentMatchers.any(Duration.class);
    }
}
