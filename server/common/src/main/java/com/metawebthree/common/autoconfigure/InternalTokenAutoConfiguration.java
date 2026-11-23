package com.metawebthree.common.autoconfigure;

import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    private static final Logger log = LoggerFactory.getLogger(InternalTokenAutoConfiguration.class);

    @Bean
    @ConditionalOnMissingBean
    public InternalTokenUtil internalTokenUtil(InternalTokenProperties properties) {
        String secret = properties.getSecret();
        if (secret == null || secret.isBlank()) {
            // Fail closed: an unconfigured secret gets a random per-boot value
            // instead of a public hard-coded default. Misconfigured services
            // will reject each other's tokens, and the warning makes the cause
            // obvious. Real deployments must set INTERNAL_TOKEN_SECRET.
            secret = UUID.randomUUID().toString();
            log.warn("internal-token.secret is not configured; using a random per-boot value. "
                + "Set INTERNAL_TOKEN_SECRET on every service (gateway and downstream) for token auth to work.");
        }
        return new InternalTokenUtil(secret, properties.getTimeoutSeconds());
    }
}
