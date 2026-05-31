package com.metawebthree.gateway.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.metawebthree.gateway.auth.GatewayAuthConfig;

/**
 * Gateway authentication and authorization configuration.
 * Centralizes all auth-related beans for the Gateway layer.
 */
@Configuration
public class GatewayAuthConfiguration {

    @Bean
    public GatewayAuthConfig gatewayAuthConfig() {
        return new GatewayAuthConfig();
    }
}