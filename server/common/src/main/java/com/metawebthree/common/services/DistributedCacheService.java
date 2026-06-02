package com.metawebthree.common.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class DistributedCacheService {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    @Autowired
    @Qualifier("caffeineCacheManager")
    private CacheManager caffeineCacheManager;

    private static final String CACHE_PREFIX = "metaweb:cache:";

    @SuppressWarnings("unchecked")
    public <T> T get(String cacheName, String key) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null) {
            Cache.ValueWrapper wrapper = l1Cache.get(key);
            if (wrapper != null) {
                return (T) wrapper.get();
            }
        }

        try {
            String redisKey = CACHE_PREFIX + cacheName + ":" + key;
            Object value = redisTemplate.opsForValue().get(redisKey);
            if (value != null) {
                if (l1Cache != null) {
                    l1Cache.put(key, value);
                }
                return (T) value;
            }
        } catch (Exception e) {
            log.warn("Redis L2 cache get failed: {}", e.getMessage());
        }

        return null;
    }

    public void put(String cacheName, String key, Object value) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null) {
            l1Cache.put(key, value);
        }

        try {
            String redisKey = CACHE_PREFIX + cacheName + ":" + key;
            redisTemplate.opsForValue().set(redisKey, value);
        } catch (Exception e) {
            log.warn("Redis L2 cache put failed: {}", e.getMessage());
        }
    }

    public void put(String cacheName, String key, Object value, long ttl, TimeUnit timeUnit) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null) {
            l1Cache.put(key, value);
        }

        try {
            String redisKey = CACHE_PREFIX + cacheName + ":" + key;
            redisTemplate.opsForValue().set(redisKey, value, ttl, timeUnit);
        } catch (Exception e) {
            log.warn("Redis L2 cache put with TTL failed: {}", e.getMessage());
        }
    }

    public void evict(String cacheName, String key) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null) {
            l1Cache.evict(key);
        }

        try {
            String redisKey = CACHE_PREFIX + cacheName + ":" + key;
            redisTemplate.delete(redisKey);
        } catch (Exception e) {
            log.warn("Redis L2 cache evict failed: {}", e.getMessage());
        }
    }

    public void clear(String cacheName) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null) {
            l1Cache.clear();
        }

        try {
            String pattern = CACHE_PREFIX + cacheName + ":*";
            var keys = redisTemplate.keys(pattern);
            if (keys != null && !keys.isEmpty()) {
                redisTemplate.delete(keys);
            }
        } catch (Exception e) {
            log.warn("Redis L2 cache clear failed: {}", e.getMessage());
        }
    }

    public boolean containsKey(String cacheName, String key) {
        Cache l1Cache = caffeineCacheManager.getCache(cacheName);
        if (l1Cache != null && l1Cache.get(key) != null) {
            return true;
        }

        try {
            String redisKey = CACHE_PREFIX + cacheName + ":" + key;
            return Boolean.TRUE.equals(redisTemplate.hasKey(redisKey));
        } catch (Exception e) {
            log.warn("Redis L2 cache containsKey failed: {}", e.getMessage());
        }

        return false;
    }
}