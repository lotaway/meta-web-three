package com.metawebthree.common.registration;

import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.generated.rpc.platform.SendEmailRequest;
import com.metawebthree.common.generated.rpc.platform.SendEmailResponse;
import com.metawebthree.common.services.DistributedCacheService;

import lombok.extern.slf4j.Slf4j;

import java.security.SecureRandom;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

@Slf4j
public class EmailVerificationCodeService {

    private static final SecureRandom RANDOM = new SecureRandom();

    private final DistributedCacheService cacheService;
    private final MessageService messageService;
    private final String cacheName;
    private final int codeLength;
    private final long ttlMinutes;
    private final boolean emailEnabled;
    private final String emailSubject;

    public EmailVerificationCodeService(DistributedCacheService cacheService,
                                        MessageService messageService,
                                        String cacheName,
                                        int codeLength,
                                        long ttlMinutes,
                                        boolean emailEnabled,
                                        String emailSubject) {
        this.cacheService = cacheService;
        this.messageService = messageService;
        this.cacheName = cacheName;
        this.codeLength = codeLength;
        this.ttlMinutes = ttlMinutes;
        this.emailEnabled = emailEnabled;
        this.emailSubject = emailSubject;
    }

    public void sendCode(String email) {
        String normalized = normalize(email);
        String code = generateCode();
        cacheService.put(cacheName, normalized, code, ttlMinutes, TimeUnit.MINUTES);
        if (emailEnabled) {
            deliver(normalized, code);
        }
    }

    public boolean verifyCode(String email, String code) {
        if (email == null || code == null) {
            return false;
        }
        String cached = cacheService.get(cacheName, normalize(email));
        cacheService.evict(cacheName, normalize(email));
        if (cached == null) {
            return false;
        }
        return cached.equals(code.trim());
    }

    private void deliver(String email, String code) {
        SendEmailResponse response;
        try {
            response = messageService.sendEmail(SendEmailRequest.newBuilder()
                    .setTo(email)
                    .setTitle(emailSubject)
                    .setContent(buildEmailContent(code))
                    .build());
        } catch (Exception e) {
            cacheService.evict(cacheName, email);
            log.warn("Email verification code delivery failed for {}: {}", email, e.getMessage());
            throw new BusinessException(ResponseStatus.EMAIL_VERIFICATION_CODE_SEND_FAILED);
        }
        if (!response.getSuccess()) {
            cacheService.evict(cacheName, email);
            log.warn("Email verification code delivery rejected for {}", email);
            throw new BusinessException(ResponseStatus.EMAIL_VERIFICATION_CODE_SEND_FAILED);
        }
    }

    private String generateCode() {
        StringBuilder code = new StringBuilder(codeLength);
        for (int i = 0; i < codeLength; i++) {
            code.append(RANDOM.nextInt(10));
        }
        return code.toString();
    }

    private String normalize(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }

    private String buildEmailContent(String code) {
        return """
            <!DOCTYPE html>
            <html>
            <head><meta charset="UTF-8"></head>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Verification Code</h2>
                <p>Your verification code is:</p>
                <h1 style="color: #1890ff; letter-spacing: 5px;">%s</h1>
                <p>This code will expire in %d minutes.</p>
                <p>If you did not request this, please ignore this email.</p>
                <hr>
                <p style="color: #999; font-size: 12px;">MetaWebThree Team</p>
            </body>
            </html>
            """.formatted(code, ttlMinutes);
    }
}