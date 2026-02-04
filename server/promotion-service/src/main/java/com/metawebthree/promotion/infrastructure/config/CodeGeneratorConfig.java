package com.metawebthree.promotion.infrastructure.config;

import java.security.SecureRandom;
import java.time.LocalDateTime;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.metawebthree.promotion.application.CouponCommandService;
import com.metawebthree.promotion.application.CouponPolicy;
import com.metawebthree.promotion.application.CouponQueryService;
import com.metawebthree.promotion.application.CouponTypeCommandService;
import com.metawebthree.promotion.application.CouponTypeQueryService;
import com.metawebthree.promotion.domain.ports.CodeGenerator;
import com.metawebthree.promotion.domain.ports.CouponBatchRepository;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

@Configuration
@EnableConfigurationProperties(PromotionProperties.class)
public class CodeGeneratorConfig {

    @Bean
    public CodeGenerator codeGenerator(PromotionProperties properties) {
        String alphabet = requireText(properties.getCodeAlphabet(), "codeAlphabet");
        int length = requirePositive(properties.getCodeLength(), "codeLength");
        return new SecureCodeGenerator(alphabet, length);
    }

    @Bean
    public TimeProvider timeProvider() {
        return LocalDateTime::now;
    }

    @Bean
    public CouponPolicy couponPolicy(PromotionProperties properties) {
        int retryLimit = requirePositive(properties.getRetryLimit(), "retryLimit");
        int maxGenerateCount = requirePositive(properties.getMaxGenerateCount(), "maxGenerateCount");
        return new CouponPolicy(retryLimit, maxGenerateCount);
    }

    @Bean
    public CouponTypeCommandService couponTypeCommandService(CouponTypeRepository couponTypeRepository,
            TimeProvider timeProvider) {
        return new CouponTypeCommandService(couponTypeRepository, timeProvider);
    }

    @Bean
    public CouponTypeQueryService couponTypeQueryService(CouponTypeRepository couponTypeRepository,
            TimeProvider timeProvider) {
        return new CouponTypeQueryService(couponTypeRepository, timeProvider);
    }

    @Bean
    public CouponCommandService couponCommandService(CouponRepository couponRepository,
            CouponTypeRepository couponTypeRepository, CouponBatchRepository couponBatchRepository,
            CodeGenerator codeGenerator, TimeProvider timeProvider, CouponPolicy policy) {
        return new CouponCommandService(couponRepository, couponTypeRepository, couponBatchRepository,
                codeGenerator, timeProvider, policy);
    }

    @Bean
    public CouponQueryService couponQueryService(CouponRepository couponRepository,
            CouponTypeRepository couponTypeRepository, TimeProvider timeProvider) {
        return new CouponQueryService(couponRepository, couponTypeRepository, timeProvider);
    }

    static class SecureCodeGenerator implements CodeGenerator {
        private final char[] alphabet;
        private final int length;
        private final SecureRandom random;

        SecureCodeGenerator(String alphabet, int length) {
            this.alphabet = alphabet.toCharArray();
            this.length = length;
            this.random = new SecureRandom();
        }

        @Override
        public String nextCode() {
            char[] buf = new char[length];
            for (int i = 0; i < length; i++) {
                buf[i] = alphabet[random.nextInt(alphabet.length)];
            }
            return new String(buf);
        }
    }

    private String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " required");
        }
        return value;
    }

    private int requirePositive(int value, String name) {
        if (value < 1) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }
}
