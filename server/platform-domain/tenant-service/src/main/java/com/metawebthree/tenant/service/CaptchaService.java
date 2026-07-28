package com.metawebthree.tenant.service;

import com.metawebthree.common.services.DistributedCacheService;

import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Service
public class CaptchaService {

    private static final String CACHE_NAME = "captcha";
    private static final long TTL_MINUTES = 5;
    private static final SecureRandom RANDOM = new SecureRandom();

    private final DistributedCacheService cacheService;

    public CaptchaService(DistributedCacheService cacheService) {
        this.cacheService = cacheService;
    }

    public CaptchaResult generate() {
        int a = RANDOM.nextInt(50) + 1;
        int b = RANDOM.nextInt(50) + 1;
        int answer = a + b;
        String token = UUID.randomUUID().toString();

        cacheService.put(CACHE_NAME, token, String.valueOf(answer), TTL_MINUTES, TimeUnit.MINUTES);

        return new CaptchaResult(token, a + " + " + b + " = ?");
    }

    public boolean verify(String token, String answer) {
        if (token == null || answer == null) {
            return false;
        }
        String cached = cacheService.get(CACHE_NAME, token);
        if (cached == null) {
            return false;
        }
        cacheService.evict(CACHE_NAME, token);
        return cached.equals(answer.trim());
    }

    public record CaptchaResult(String token, String question) {}
}
