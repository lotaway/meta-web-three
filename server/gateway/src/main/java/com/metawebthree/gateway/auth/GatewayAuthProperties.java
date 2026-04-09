package com.metawebthree.gateway.auth;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gateway.auth")
public record GatewayAuthProperties(
        String authorizationHeader,
        String tokenPrefix,
        String protectedPathPrefix,
        List<String> excludedPathPrefixes,
        List<String> excludedPathKeywords) {
}
