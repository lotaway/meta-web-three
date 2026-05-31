package com.metawebthree.gateway.auth;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.Ordered;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.server.ServerWebExchange;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import reactor.core.publisher.Mono;

/**
 * Role-based access control filter that enforces permission checks at the Gateway level.
 * Consolidates authorization logic from all services into a unified layer.
 */
@Component
public class RoleBasedAccessFilter implements GlobalFilter, Ordered {

    private final GatewayAuthConfig authConfig;
    private final GatewayAuthProperties authProperties;
    private final ObjectMapper objectMapper;
    private final AntPathMatcher pathMatcher;

    private static final String USER_ROLE_HEADER = "X-User-Role";

    public RoleBasedAccessFilter(GatewayAuthConfig authConfig, GatewayAuthProperties authProperties, ObjectMapper objectMapper) {
        this.authConfig = authConfig;
        this.authProperties = authProperties;
        this.objectMapper = objectMapper;
        this.pathMatcher = new AntPathMatcher();
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String path = exchange.getRequest().getPath().value();
        String userRole = exchange.getRequest().getHeaders().getFirst(USER_ROLE_HEADER);

        // Skip RBAC for paths without role requirements
        if (userRole == null || shouldSkipRbac(path)) {
            return chain.filter(exchange);
        }

        // Check if user role has access to this route
        if (!authConfig.hasRoleForRoute(userRole, path)) {
            return writeForbidden(exchange, "ROLE_ACCESS_DENIED", 
                "Insufficient permissions for this resource");
        }

        return chain.filter(exchange);
    }

    private boolean shouldSkipRbac(String path) {
        List<String> patterns = authProperties.rbacExcludedPathPatterns();
        return patterns != null && patterns.stream().anyMatch(pattern -> pathMatcher.match(pattern, path));
    }

    private Mono<Void> writeForbidden(ServerWebExchange exchange, String errorCode, String message) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(HttpStatus.FORBIDDEN);
        response.getHeaders().setContentType(MediaType.APPLICATION_JSON);
        String payload = toJson(Map.of(
            "code", errorCode,
            "message", message,
            "path", exchange.getRequest().getPath().value(),
            "timestamp", Instant.now().toString()));
        DataBuffer buffer = response.bufferFactory().wrap(payload.getBytes(StandardCharsets.UTF_8));
        return response.writeWith(Mono.just(buffer));
    }

    private String toJson(Map<String, String> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            return "{\"code\":\"ROLE_RESPONSE_ERROR\",\"message\":\"Authorization failed\"}";
        }
    }

    @Override
    public int getOrder() {
        return -90; // Run after UserAuthFilter (-100)
    }
}