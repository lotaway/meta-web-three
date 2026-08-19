package com.metawebthree.developerportal.service;

import com.metawebthree.common.services.DistributedCacheService;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CaptchaServiceTest {

    @Mock
    private DistributedCacheService cacheService;

    @Test
    void testGenerate_StoresAnswerWithTtl() {
        CaptchaService captchaService = new CaptchaService(cacheService);

        CaptchaService.CaptchaResult result = captchaService.generate();

        assertNotNull(result.token());
        assertTrue(result.question().contains("= ?"));
        verify(cacheService).put(
            eq("developer_captcha"),
            eq(result.token()),
            anyString(),
            eq(5L),
            eq(TimeUnit.MINUTES));
    }

    @Test
    void testVerify_CorrectAnswer_ConsumesToken() {
        CaptchaService captchaService = new CaptchaService(cacheService);
        when(cacheService.get(eq("developer_captcha"), eq("token-1"))).thenReturn("7");

        boolean valid = captchaService.verify("token-1", " 7 ");

        assertTrue(valid);
        verify(cacheService).evict("developer_captcha", "token-1");
    }

    @Test
    void testVerify_WrongAnswer_ConsumesTokenAndReturnsFalse() {
        CaptchaService captchaService = new CaptchaService(cacheService);
        when(cacheService.get(eq("developer_captcha"), eq("token-2"))).thenReturn("7");

        boolean valid = captchaService.verify("token-2", "8");

        assertFalse(valid);
        verify(cacheService).evict("developer_captcha", "token-2");
    }

    @Test
    void testVerify_NullInputs_ReturnsFalse() {
        CaptchaService captchaService = new CaptchaService(cacheService);

        assertFalse(captchaService.verify(null, "7"));
        assertFalse(captchaService.verify("token-3", null));
        verify(cacheService, never()).evict(anyString(), anyString());
    }
}