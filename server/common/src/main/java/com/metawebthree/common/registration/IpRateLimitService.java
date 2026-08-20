package com.metawebthree.common.registration;

import com.metawebthree.common.web.ClientIpResolver;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.data.redis.core.RedisTemplate;

import java.time.Duration;

public class IpRateLimitService {

    private final RedisTemplate<String, Object> redisTemplate;
    private final String keyPrefix;
    private final int maxRequests;
    private final Duration refreshPeriod;

    public IpRateLimitService(RedisTemplate<String, Object> redisTemplate,
                              String keyPrefix,
                              int maxRequests,
                              Duration refreshPeriod) {
        this.redisTemplate = redisTemplate;
        this.keyPrefix = keyPrefix;
        this.maxRequests = maxRequests;
        this.refreshPeriod = refreshPeriod;
    }

    public boolean isAllowed(HttpServletRequest request) {
        String key = keyPrefix + ClientIpResolver.resolve(request);
        Long count = redisTemplate.opsForValue().increment(key);
        if (count != null && count == 1) {
            redisTemplate.expire(key, refreshPeriod);
        }
        return count != null && count <= maxRequests;
    }
}