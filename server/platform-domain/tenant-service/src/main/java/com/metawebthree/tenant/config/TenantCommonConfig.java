package com.metawebthree.tenant.config;

import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.RedisCacheConfig;
import com.metawebthree.common.exception.GlobalExceptionHandler;
import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.registration.EmailVerificationCodeService;
import com.metawebthree.common.registration.IpRateLimitService;
import com.metawebthree.common.registration.TokenCaptchaService;
import com.metawebthree.common.services.DistributedCacheService;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.data.redis.core.RedisTemplate;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Configuration
@Import({
    RedisCacheConfig.class,
    DistributedCacheService.class,
    GlobalExceptionHandler.class,
    CommonSecurityConfig.class
})
public class TenantCommonConfig {

    @DubboReference(check = false, lazy = true)
    private MessageService messageService;

    @Bean
    public TokenCaptchaService tenantCaptchaService(DistributedCacheService cacheService) {
        return new TokenCaptchaService(cacheService, "captcha", 5, TimeUnit.MINUTES);
    }

    @Bean
    public EmailVerificationCodeService tenantEmailVerificationService(
            DistributedCacheService cacheService,
            @Value("${notification.email.enabled:true}") boolean emailEnabled) {
        return new EmailVerificationCodeService(cacheService, messageService,
                "email_verification", 6, 10, emailEnabled,
                "[MetaWebThree] Email Verification Code");
    }

    @Bean
    public IpRateLimitService tenantCaptchaRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                       @Value("${service-governance.rate-limiter.tenantCaptcha.limit-for-period:10}") int maxRequests,
                                                       @Value("${service-governance.rate-limiter.tenantCaptcha.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:tenant:captcha:",
                maxRequests, refreshPeriod);
    }

    @Bean
    public IpRateLimitService tenantEmailRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                     @Value("${service-governance.rate-limiter.tenantEmailCode.limit-for-period:5}") int maxRequests,
                                                     @Value("${service-governance.rate-limiter.tenantEmailCode.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:tenant:email:",
                maxRequests, refreshPeriod);
    }

    @Bean
    public IpRateLimitService tenantRegisterRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                        @Value("${service-governance.rate-limiter.tenantRegister.limit-for-period:3}") int maxRequests,
                                                        @Value("${service-governance.rate-limiter.tenantRegister.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:tenant:register:",
                maxRequests, refreshPeriod);
    }
}