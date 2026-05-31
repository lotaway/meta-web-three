package com.metawebthree.gateway.auth;

import java.util.List;
import java.util.Map;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Gateway authentication properties that can be configured via application.yml.
 * Supports both basic JWT validation and enhanced role-based access control.
 */
@ConfigurationProperties(prefix = "gateway.auth")
public record GatewayAuthProperties(
        String authorizationHeader,
        String tokenPrefix,
        String protectedPathPrefix,
        List<String> excludedPathPatterns,
        List<String> excludedPathKeywords,
        List<String> rbacExcludedPathPatterns,
        Map<String, List<String>> routeRoles,
        Map<String, List<String>> rolePermissions) {

    public GatewayAuthProperties() {
        this("Authorization", "Bearer ", "/", List.of(), List.of(), List.of(), Map.of(), Map.of());
    }
}