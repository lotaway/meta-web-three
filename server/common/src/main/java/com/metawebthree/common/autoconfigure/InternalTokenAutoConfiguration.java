package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import com.metawebthree.common.utils.InternalTokenUtil;

/**
 * Registers the shared HMAC token utility for every service that depends on the
 * common module, including the reactive gateway (the token is generated there).
 *
 * <p>This replaces the old practice of listing {@code com.metawebthree.common.*}
 * subpackages in every application's {@code scanBasePackages}, or manually
 * constructing the utility in a service configuration class. Both approaches
 * silently produce different secrets/timeouts when one application is missed.</p>
 */
@AutoConfiguration
@EnableConfigurationProperties(InternalTokenProperties.class)
public class InternalTokenAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public InternalTokenUtil internalTokenUtil(InternalTokenProperties properties) {
        return new InternalTokenUtil(properties.getSecret(), properties.getTimeoutSeconds());
    }
}
