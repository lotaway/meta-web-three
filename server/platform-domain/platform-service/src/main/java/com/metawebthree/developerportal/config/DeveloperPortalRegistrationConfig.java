package com.metawebthree.developerportal.config;

import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.registration.EmailVerificationCodeService;
import com.metawebthree.common.registration.IpRateLimitService;
import com.metawebthree.common.registration.TokenCaptchaService;
import com.metawebthree.common.services.DistributedCacheService;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.RedisTemplate;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Configuration
public class DeveloperPortalRegistrationConfig {

    @DubboReference(check = false, lazy = true)
    private MessageService messageService;

    @Bean
    public TokenCaptchaService developerCaptchaService(DistributedCacheService cacheService) {
        return new TokenCaptchaService(cacheService, "developer_captcha", 5, TimeUnit.MINUTES);
    }

    @Bean
    public EmailVerificationCodeService developerEmailVerificationService(
            DistributedCacheService cacheService,
            @Value("${notification.email.enabled:true}") boolean emailEnabled) {
        return new EmailVerificationCodeService(cacheService, messageService,
                "developer_email_verification", 6, 10, emailEnabled,
                "[MetaWebThree] Developer Registration Verification Code");
    }

    @Bean
    public IpRateLimitService developerCaptchaRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                          @Value("${service-governance.rate-limiter.developerCaptcha.limit-for-period:10}") int maxRequests,
                                                          @Value("${service-governance.rate-limiter.developerCaptcha.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:developer:captcha:",
                maxRequests, refreshPeriod);
    }

    @Bean
    public IpRateLimitService developerEmailRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                        @Value("${service-governance.rate-limiter.developerEmailCode.limit-for-period:5}") int maxRequests,
                                                        @Value("${service-governance.rate-limiter.developerEmailCode.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:developer:email:",
                maxRequests, refreshPeriod);
    }

    @Bean
    public IpRateLimitService developerRegisterRateLimiter(RedisTemplate<String, Object> redisTemplate,
                                                           @Value("${service-governance.rate-limiter.developerRegister.limit-for-period:3}") int maxRequests,
                                                           @Value("${service-governance.rate-limiter.developerRegister.limit-refresh-period:60s}") Duration refreshPeriod) {
        return new IpRateLimitService(redisTemplate, "metaweb:ratelimit:developer:register:",
                maxRequests, refreshPeriod);
    }
}