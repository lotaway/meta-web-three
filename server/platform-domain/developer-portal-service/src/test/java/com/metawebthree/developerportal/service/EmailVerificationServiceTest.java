package com.metawebthree.developerportal.service;

import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.generated.rpc.platform.SendEmailResponse;
import com.metawebthree.common.services.DistributedCacheService;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EmailVerificationServiceTest {

    private static final String CACHE_NAME = "developer_email_verification";

    @Mock
    private DistributedCacheService cacheService;

    @Mock
    private MessageService messageService;

    private EmailVerificationService emailVerificationService;

    @BeforeEach
    void setUp() {
        emailVerificationService = new EmailVerificationService(cacheService);
        ReflectionTestUtils.setField(emailVerificationService, "messageService", messageService);
        ReflectionTestUtils.setField(emailVerificationService, "emailEnabled", false);
    }

    @Test
    void testSendCode_StoresSixDigitCode() {
        boolean sent = emailVerificationService.sendCode("dev@example.com");

        assertTrue(sent);
        verify(cacheService).put(
            eq(CACHE_NAME),
            eq("dev@example.com"),
            anyString(),
            eq(10L),
            eq(TimeUnit.MINUTES));
    }

    @Test
    void testSendCode_DeliveryFailure_EvictsCode() {
        ReflectionTestUtils.setField(emailVerificationService, "emailEnabled", true);
        when(messageService.sendEmail(any()))
                .thenThrow(new org.apache.dubbo.rpc.RpcException("SMTP down"));

        boolean sent = emailVerificationService.sendCode("dev@example.com");

        assertFalse(sent);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testSendCode_Success_DoesNotEvict() {
        ReflectionTestUtils.setField(emailVerificationService, "emailEnabled", true);
        when(messageService.sendEmail(any()))
                .thenReturn(SendEmailResponse.newBuilder().setSuccess(true).build());

        boolean sent = emailVerificationService.sendCode("dev@example.com");

        assertTrue(sent);
        verify(cacheService, never()).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testSendCode_ResponseFailure_EvictsCode() {
        ReflectionTestUtils.setField(emailVerificationService, "emailEnabled", true);
        when(messageService.sendEmail(any()))
                .thenReturn(SendEmailResponse.newBuilder().setSuccess(false).build());

        boolean sent = emailVerificationService.sendCode("dev@example.com");

        assertFalse(sent);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_CorrectCode_ConsumesEntry() {
        when(cacheService.get(eq(CACHE_NAME), eq("dev@example.com"))).thenReturn("123456");

        boolean valid = emailVerificationService.verifyCode("dev@example.com", " 123456 ");

        assertTrue(valid);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_WrongCode_ConsumesEntryAndReturnsFalse() {
        when(cacheService.get(eq(CACHE_NAME), eq("dev@example.com"))).thenReturn("123456");

        boolean valid = emailVerificationService.verifyCode("dev@example.com", "654321");

        assertFalse(valid);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_NullInputs_ReturnsFalse() {
        assertFalse(emailVerificationService.verifyCode(null, "123456"));
        assertFalse(emailVerificationService.verifyCode("dev@example.com", null));
        verify(cacheService, never()).evict(anyString(), anyString());
    }
}