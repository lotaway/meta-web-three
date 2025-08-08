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
import org.web3j.tuples.generated.Tuple2;

import com.metawebthree.common.contants.RequestHeaderKeys;
import com.metawebthree.common.utils.UserJwtUtil;

import io.jsonwebtoken.Claims;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

@Slf4j
@Component
public class UserAuthFilter implements GlobalFilter, Ordered {

    private final static String AUTH_HEADER = "Bearer ";

    private final UserJwtUtil userJwtUtil;

    UserAuthFilter(UserJwtUtil userJwtUtil) {
        this.userJwtUtil = userJwtUtil;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getPath().value();

        // @TODO add more paths to exclude from authentication
        if (!path.startsWith("/user-service/") || path.startsWith("/user-/signIn") || path.startsWith("/user/create") ||
                path.startsWith("/user/checkWeb3SignerMessage") || path.startsWith("/actuator")) {
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
        Tuple2<Boolean, Claims> validResult = validateToken(token);
        Boolean isValid = validResult.component1();
        Claims claims = validResult.component2();

        if (!isValid) {
            ServerHttpResponse response = exchange.getResponse();
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return response.setComplete();
        }

        ServerWebExchange _exchange = exchange.mutate()
                .request(
                        exchange.getRequest().mutate()
                                .header(RequestHeaderKeys.USER_ID.getValue(), userJwtUtil.getUserId(claims).toString())
                                .header(RequestHeaderKeys.USER_NAME.getValue(),
                                        userJwtUtil.getUserName(claims).toString())
                                .header(RequestHeaderKeys.USER_ROLE.getValue(),
                                        userJwtUtil.getUserRole(claims).toString())
                                .build())
                .build();
        return chain.filter(_exchange);
    }

    private Tuple2<Boolean, Claims> validateToken(String token) {
        Optional<Claims> oClaims = userJwtUtil.tryDecode(token);
        if (oClaims.isEmpty()) {
            return new Tuple2<>(false, null);
        }
        Claims claims = oClaims.get();
        Long userId = userJwtUtil.getUserId(claims);
        if (userId == null) {
            return new Tuple2<>(false, null);
        }
        if (userJwtUtil.isTokenExpired(claims.getExpiration())) {
            return new Tuple2<>(false, null);
        }
        return new Tuple2<>(true, claims);
    }

    @Override
    public int getOrder() {
        return -100;
    }
}