package com.metawebthree.developerportal.service;

import org.apache.dubbo.config.annotation.DubboReference;

import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.generated.rpc.platform.SendEmailRequest;
import com.metawebthree.common.generated.rpc.platform.SendEmailResponse;
import com.metawebthree.common.services.DistributedCacheService;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.util.concurrent.TimeUnit;

@Service
public class EmailVerificationService {

    private static final String CACHE_NAME = "developer_email_verification";
    private static final long TTL_MINUTES = 10;
    private static final int CODE_LENGTH = 6;
    private static final SecureRandom RANDOM = new SecureRandom();

    private final DistributedCacheService cacheService;

    @DubboReference(check = false, lazy = true)
    private MessageService messageService;

    @Value("${notification.email.enabled:true}")
    private boolean emailEnabled;

    public EmailVerificationService(DistributedCacheService cacheService) {
        this.cacheService = cacheService;
    }

    public boolean sendCode(String email) {
        String code = generateCode();
        cacheService.put(CACHE_NAME, email, code, TTL_MINUTES, TimeUnit.MINUTES);

        if (!emailEnabled) {
            return true;
        }

        try {
            SendEmailRequest request = SendEmailRequest.newBuilder()
                    .setTo(email)
                    .setTitle("[MetaWebThree] Developer Registration Verification Code")
                    .setContent(buildEmailContent(code))
                    .build();
            SendEmailResponse response = messageService.sendEmail(request);
            if (response == null || !response.getSuccess()) {
                cacheService.evict(CACHE_NAME, email);
                return false;
            }
            return true;
        } catch (Exception e) {
            cacheService.evict(CACHE_NAME, email);
            return false;
        }
    }

    public boolean verifyCode(String email, String code) {
        if (email == null || code == null) {
            return false;
        }
        String cached = cacheService.get(CACHE_NAME, email);
        if (cached == null) {
            return false;
        }
        cacheService.evict(CACHE_NAME, email);
        return cached.equals(code.trim());
    }

    private String generateCode() {
        StringBuilder sb = new StringBuilder(CODE_LENGTH);
        for (int i = 0; i < CODE_LENGTH; i++) {
            sb.append(RANDOM.nextInt(10));
        }
        return sb.toString();
    }

    private String buildEmailContent(String code) {
        return """
            <!DOCTYPE html>
            <html>
            <head><meta charset="UTF-8"></head>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Developer Registration Verification</h2>
                <p>Your verification code is:</p>
                <h1 style="color: #1890ff; letter-spacing: 5px;">%s</h1>
                <p>This code will expire in %d minutes.</p>
                <p>If you did not request this, please ignore this email.</p>
                <hr>
                <p style="color: #999; font-size: 12px;">MetaWebThree Team</p>
            </body>
            </html>
            """.formatted(code, TTL_MINUTES);
    }
}