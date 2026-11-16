package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Import;
import org.springframework.data.redis.core.RedisTemplate;

import com.metawebthree.common.config.RedisCacheConfig;
import com.metawebthree.common.services.DistributedCacheService;

/**
 * Custom Redis serializer / cache manager wiring. Boot's default Redis
 * auto-configuration is left alone unless a service explicitly opts in.
 */
@AutoConfiguration(before = org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration.class)
@ConditionalOnClass(RedisTemplate.class)
@ConditionalOnProperty(prefix = "meta.common.storage.redis", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
@Import({RedisCacheConfig.class, DistributedCacheService.class})
public class RedisCacheAutoConfiguration {
}
