package com.metawebthree.Filters;

import java.util.Optional;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;

import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.utils.UserJwtUtil;

import io.jsonwebtoken.Claims;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

@Slf4j
@Component
public class UserAuthFilter implements GlobalFilter, Ordered {

    private static final String AUTH_HEADER = "Bearer ";

    private final UserJwtUtil userJwtUtil;

    UserAuthFilter(UserJwtUtil userJwtUtil) {
        this.userJwtUtil = userJwtUtil;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getPath().value();

        // @TODO add more paths to exclude from authentication
        if (path.contains("/v3/api-docs")
                || path.contains("/swagger-ui")
                || !path.startsWith("/user-service/")
                || path.startsWith("/user-service/user/signIn")
                || path.startsWith("/user-service/user/create")
                || path.startsWith("/user-service/user/checkWeb3SignerMessage")
                || path.startsWith("/actuator")) {
            return chain.filter(exchange);
        }
        log.info("matching Authorization path: " + path);
        String authHeader = request.getHeaders().getFirst("Authorization");
        if (authHeader == null || !authHeader.startsWith(AUTH_HEADER)) {
            ServerHttpResponse response = exchange.getResponse();
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return response.setComplete();
        }

        String token = authHeader.substring(AUTH_HEADER.length());
        ValidateTokenResponse validResult = validateToken(token);

        if (!validResult.isValid) {
            ServerHttpResponse response = exchange.getResponse();
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return response.setComplete();
        }

        ServerWebExchange _exchange = exchange.mutate()
                .request(
                        exchange.getRequest().mutate()
                                .header(RequestHeaderKeys.USER_ID.getValue(),
                                        userJwtUtil.getUserId(validResult.claims).toString())
                                .header(RequestHeaderKeys.USER_NAME.getValue(),
                                        userJwtUtil.getUserName(validResult.claims).toString())
                                .header(RequestHeaderKeys.USER_ROLE.getValue(),
                                        userJwtUtil.getUserRole(validResult.claims).toString())
                                .build())
                .build();
        return chain.filter(_exchange);
    }

    private ValidateTokenResponse validateToken(String token) {
        Optional<Claims> oClaims = userJwtUtil.tryDecode(token);
        var response = new ValidateTokenResponse();
        response.isValid = false;
        if (oClaims.isEmpty()) {
            return response;
        }
        response.claims = oClaims.get();
        Long userId = userJwtUtil.getUserId(response.claims);
        if (userId == null || userJwtUtil.isTokenExpired(response.claims.getExpiration())) {
            return response;
        }
        response.isValid = true;
        return response;
    }

    static class ValidateTokenResponse {
        public boolean isValid;
        public Claims claims;
    }

    @Override
    public int getOrder() {
        return -100;
    }
}