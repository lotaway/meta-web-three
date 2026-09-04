package com.metawebthree.common.registration;

import com.metawebthree.common.services.DistributedCacheService;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class TokenCaptchaServiceTest {

    private static final String CACHE_NAME = "test_captcha";

    @Mock
    private DistributedCacheService cacheService;

    private TokenCaptchaService captchaService;

    @BeforeEach
    void setUp() {
        captchaService = new TokenCaptchaService(cacheService, CACHE_NAME, 5L, TimeUnit.MINUTES);
    }

    @Test
    void testGenerate_ReturnsTokenAndImage() {
        TokenCaptchaService.CaptchaChallenge challenge = captchaService.generate();

        assertNotNull(challenge.token());
        assertNotNull(challenge.image());
        assertTrue(challenge.image().startsWith("data:image/png;base64,"));
        verify(cacheService).put(org.mockito.ArgumentMatchers.eq(CACHE_NAME),
                org.mockito.ArgumentMatchers.eq(challenge.token()),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.eq(5L),
                org.mockito.ArgumentMatchers.eq(TimeUnit.MINUTES));
    }

    @Test
    void testVerify_CorrectAnswer_ConsumesToken() {
        when(cacheService.get(CACHE_NAME, "token-1")).thenReturn("K7M2");

        boolean valid = captchaService.verify("token-1", " k7m2 ");

        assertTrue(valid);
        verify(cacheService).evict(CACHE_NAME, "token-1");
    }

    @Test
    void testVerify_WrongAnswer_ConsumesTokenAndReturnsFalse() {
        when(cacheService.get(CACHE_NAME, "token-2")).thenReturn("K7M2");

        boolean valid = captchaService.verify("token-2", "ZZZZ");

        assertFalse(valid);
        verify(cacheService).evict(CACHE_NAME, "token-2");
    }

    @Test
    void testVerify_UnknownOrNullToken_ReturnsFalse() {
        when(cacheService.get(CACHE_NAME, "token-3")).thenReturn(null);

        assertFalse(captchaService.verify("token-3", "K7M2"));
        assertFalse(captchaService.verify(null, "K7M2"));
        assertFalse(captchaService.verify("token-4", null));

        verify(cacheService).evict(CACHE_NAME, "token-3");
        verify(cacheService, never()).evict(CACHE_NAME, "token-4");
        verify(cacheService, never()).evict(CACHE_NAME, null);
    }
}