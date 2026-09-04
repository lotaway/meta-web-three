package com.metawebthree.common.registration;

import jakarta.servlet.http.HttpServletRequest;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class IpRateLimitServiceTest {

    private static final int MAX_REQUESTS = 3;
    private static final Duration REFRESH_PERIOD = Duration.ofSeconds(60);
    private static final String KEY_PREFIX = "metaweb:ratelimit:test:";

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
        rateLimitService = new IpRateLimitService(redisTemplate, KEY_PREFIX, MAX_REQUESTS, REFRESH_PERIOD);
    }

    @Test
    void testIsAllowed_TrustedHeaderFirst() {
        when(request.getHeader("X-Client-IP")).thenReturn("10.1.2.3");
        when(valueOperations.increment("metaweb:ratelimit:test:10.1.2.3")).thenReturn(3L);

        assertTrue(rateLimitService.isAllowed(request));
    }

    @Test
    void testIsAllowed_IgnoresSpoofedForwardedHeaders() {
        when(request.getHeader("X-Client-IP")).thenReturn("10.1.2.3");
        when(valueOperations.increment("metaweb:ratelimit:test:10.1.2.3")).thenReturn(1L);

        rateLimitService.isAllowed(request);

        verify(valueOperations).increment("metaweb:ratelimit:test:10.1.2.3");
        verify(valueOperations, never()).increment(eq("metaweb:ratelimit:test:9.9.9.9"));
    }

    @Test
    void testIsAllowed_FallsBackToRemoteAddress() {
        when(request.getHeader("X-Client-IP")).thenReturn(null);
        when(request.getRemoteAddr()).thenReturn("192.168.1.10");
        when(valueOperations.increment("metaweb:ratelimit:test:192.168.1.10")).thenReturn(1L);

        assertTrue(rateLimitService.isAllowed(request));
        verify(redisTemplate).expire("metaweb:ratelimit:test:192.168.1.10", REFRESH_PERIOD);
    }

    @Test
    void testIsAllowed_OverLimitBlocked() {
        when(request.getHeader("X-Client-IP")).thenReturn("10.1.2.3");
        when(valueOperations.increment("metaweb:ratelimit:test:10.1.2.3")).thenReturn(4L);

        assertFalse(rateLimitService.isAllowed(request));
    }

    @Test
    void testIsAllowed_RedisFailureBlocked() {
        when(request.getHeader("X-Client-IP")).thenReturn("10.1.2.3");
        when(valueOperations.increment(eq("metaweb:ratelimit:test:10.1.2.3"))).thenReturn(null);

        assertFalse(rateLimitService.isAllowed(request));
        verify(redisTemplate, never()).expire(anyString(), any(Duration.class));
    }
}