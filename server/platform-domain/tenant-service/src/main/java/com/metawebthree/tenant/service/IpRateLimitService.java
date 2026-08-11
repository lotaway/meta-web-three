package com.metawebthree.tenant.service;

import com.metawebthree.common.constants.HeaderConstants;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class IpRateLimitService {

    private static final String KEY_PREFIX = "metaweb:ratelimit:tenant:";

    private final RedisTemplate<String, Object> redisTemplate;

    @Value("${service-governance.rate-limiter.tenantRegister.limit-for-period:3}")
    private int maxRequests;

    @Value("${service-governance.rate-limiter.tenantRegister.limit-refresh-period:60s}")
    private Duration refreshPeriod;

    public IpRateLimitService(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public boolean isAllowed(HttpServletRequest request) {
        String ip = resolveClientIp(request);
        String key = KEY_PREFIX + ip;
        Long count = redisTemplate.opsForValue().increment(key);
        if (count != null && count == 1) {
            redisTemplate.expire(key, refreshPeriod);
        }
        return count == null || count <= maxRequests;
    }

    private String resolveClientIp(HttpServletRequest request) {
        String forwarded = request.getHeader("X-Forwarded-For");
        if (forwarded != null && !forwarded.isBlank() && !"unknown".equalsIgnoreCase(forwarded)) {
            return forwarded.split(",")[0].trim();
        }
        String realIp = request.getHeader("X-Real-IP");
        if (realIp != null && !realIp.isBlank()) {
            return realIp;
        }
        return request.getRemoteAddr();
    }
}
