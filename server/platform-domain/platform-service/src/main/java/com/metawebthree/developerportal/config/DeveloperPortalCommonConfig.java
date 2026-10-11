package com.metawebthree.developerportal.config;

import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.RedisCacheConfig;
import com.metawebthree.common.exception.GlobalExceptionHandler;
import com.metawebthree.common.services.DistributedCacheService;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;

@Configuration
@Import({
    RedisCacheConfig.class,
    DistributedCacheService.class,
    GlobalExceptionHandler.class,
    CommonSecurityConfig.class
})
public class DeveloperPortalCommonConfig {
}