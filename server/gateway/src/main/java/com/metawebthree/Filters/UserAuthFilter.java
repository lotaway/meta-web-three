package com.metawebthree.Filters;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Map;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.Ordered;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;

import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.gateway.auth.GatewayAuthProperties;
import com.metawebthree.gateway.auth.UserTokenClaims;
import com.metawebthree.gateway.auth.UserTokenValidator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import reactor.core.publisher.Mono;

@Component
public class UserAuthFilter implements GlobalFilter, Ordered {

    private final UserTokenValidator tokenValidator;
    private final GatewayAuthProperties authProperties;
    private final ObjectMapper objectMapper;

    UserAuthFilter(UserTokenValidator tokenValidator, GatewayAuthProperties authProperties, ObjectMapper objectMapper) {
        this.tokenValidator = tokenValidator;
        this.authProperties = authProperties;
        this.objectMapper = objectMapper;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        if (shouldSkipAuthentication(exchange.getRequest().getPath().value())) {
            return chain.filter(exchange);
        }
        String authHeader = exchange.getRequest().getHeaders().getFirst(authProperties.authorizationHeader());
        String token = resolveToken(authHeader);
        if (token == null) {
            return writeUnauthorized(exchange, "AUTH_HEADER_INVALID");
        }
        UserTokenClaims tokenClaims = tokenValidator.validate(token);
        if (tokenClaims == null) {
            return writeUnauthorized(exchange, "AUTH_TOKEN_INVALID");
        }
        return chain.filter(addUserHeaders(exchange, tokenClaims));
    }

    private boolean shouldSkipAuthentication(String path) {
        if (!path.startsWith(authProperties.protectedPathPrefix())) {
            return true;
        }
        return authProperties.excludedPathPrefixes().stream().anyMatch(path::startsWith)
                || authProperties.excludedPathKeywords().stream().anyMatch(path::contains);
    }

    private String resolveToken(String authHeader) {
        if (authHeader == null) {
            return null;
        }
        String tokenPrefix = authProperties.tokenPrefix();
        if (!authHeader.startsWith(tokenPrefix)) {
            return null;
        }
        return authHeader.substring(tokenPrefix.length());
    }

    private ServerWebExchange addUserHeaders(ServerWebExchange exchange, UserTokenClaims tokenClaims) {
        return exchange.mutate().request(exchange.getRequest().mutate()
                .header(RequestHeaderKeys.USER_ID.getValue(), String.valueOf(tokenClaims.userId()))
                .header(RequestHeaderKeys.USER_NAME.getValue(), tokenClaims.userName())
                .header(RequestHeaderKeys.USER_ROLE.getValue(), tokenClaims.userRole())
                .build()).build();
    }

    private Mono<Void> writeUnauthorized(ServerWebExchange exchange, String errorCode) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(HttpStatus.UNAUTHORIZED);
        response.getHeaders().setContentType(MediaType.APPLICATION_JSON);
        String payload = toJson(Map.of(
                "code", errorCode,
                "message", "Authentication failed",
                "path", exchange.getRequest().getPath().value(),
                "timestamp", Instant.now().toString()));
        DataBuffer buffer = response.bufferFactory().wrap(payload.getBytes(StandardCharsets.UTF_8));
        return response.writeWith(Mono.just(buffer));
    }

    private String toJson(Map<String, String> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            return "{\"code\":\"AUTH_RESPONSE_SERIALIZATION_ERROR\",\"message\":\"Authentication failed\"}";
        }
    }

    @Override
    public int getOrder() {
        return -100;
    }
}