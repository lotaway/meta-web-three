package com.metawebthree.common.registration;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.generated.rpc.platform.SendEmailRequest;
import com.metawebthree.common.generated.rpc.platform.SendEmailResponse;
import com.metawebthree.common.services.DistributedCacheService;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EmailVerificationCodeServiceTest {

    private static final String CACHE_NAME = "test_email_verification";

    @Mock
    private DistributedCacheService cacheService;

    @Mock
    private MessageService messageService;

    private EmailVerificationCodeService service;

    @BeforeEach
    void setUp() {
        service = new EmailVerificationCodeService(cacheService, messageService,
                CACHE_NAME, 6, 10L, false, "Test Verification Code");
    }

    @Test
    void testSendCode_StoresSixDigitCode() {
        service.sendCode("Dev@Example.com");

        verify(cacheService).put(eq(CACHE_NAME), eq("dev@example.com"),
                org.mockito.ArgumentMatchers.anyString(),
                eq(10L), eq(TimeUnit.MINUTES));
        verify(messageService, never()).sendEmail(any());
    }

    @Test
    void testSendCode_DeliveryFailure_ThrowsBusinessException() {
        EmailVerificationCodeService enabledService =
                new EmailVerificationCodeService(cacheService, messageService,
                        CACHE_NAME, 6, 10L, true, "Test Verification Code");
        when(messageService.sendEmail(any()))
                .thenThrow(new org.apache.dubbo.rpc.RpcException("SMTP down"));

        BusinessException exception = assertThrows(BusinessException.class,
                () -> enabledService.sendCode("dev@example.com"));

        assertEquals("1205", exception.getStatus().getCode());
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testSendCode_RejectedByMailer_ThrowsBusinessException() {
        EmailVerificationCodeService enabledService =
                new EmailVerificationCodeService(cacheService, messageService,
                        CACHE_NAME, 6, 10L, true, "Test Verification Code");
        when(messageService.sendEmail(any()))
                .thenReturn(SendEmailResponse.newBuilder().setSuccess(false).build());

        BusinessException exception = assertThrows(BusinessException.class,
                () -> enabledService.sendCode("dev@example.com"));

        assertEquals("1205", exception.getStatus().getCode());
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_CorrectCode_ConsumesEntry() {
        when(cacheService.get(eq(CACHE_NAME), eq("dev@example.com"))).thenReturn("123456");

        boolean valid = service.verifyCode("Dev@Example.com", " 123456 ");

        assertTrue(valid);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_WrongCode_ConsumesEntryAndReturnsFalse() {
        when(cacheService.get(eq(CACHE_NAME), eq("dev@example.com"))).thenReturn("123456");

        boolean valid = service.verifyCode("dev@example.com", "654321");

        assertFalse(valid);
        verify(cacheService).evict(CACHE_NAME, "dev@example.com");
    }

    @Test
    void testVerifyCode_NullInputs_ReturnsFalse() {
        assertFalse(service.verifyCode(null, "123456"));
        assertFalse(service.verifyCode("dev@example.com", null));
        verify(cacheService, never()).evict(org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.anyString());
    }
}