package com.metawebthree.Filters;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.server.ServerWebExchange;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.utils.InternalTokenUtil;
import com.metawebthree.gateway.auth.GatewayAuthProperties;
import com.metawebthree.gateway.auth.UserTokenClaims;
import com.metawebthree.gateway.auth.UserTokenValidator;

import reactor.core.publisher.Mono;

class UserAuthFilterWebSocketTest {

    private static final String WS_PATH = "/factory-service/ws/digital-twin";

    @Test
    void validatesWsQueryTokenAndStripsItBeforeProxying() {
        UserTokenValidator validator = Mockito.mock(UserTokenValidator.class);
        Mockito.when(validator.validate("jwt-1"))
                .thenReturn(new UserTokenClaims(7L, "digital-twin", "ADMIN", null));
        InternalTokenUtil internalTokenUtil = Mockito.mock(InternalTokenUtil.class);
        Mockito.when(internalTokenUtil.generate()).thenReturn("internal-token");

        GatewayFilterChain chain = Mockito.mock(GatewayFilterChain.class);
        ArgumentCaptor<ServerWebExchange> captor = ArgumentCaptor.forClass(ServerWebExchange.class);
        Mockito.when(chain.filter(captor.capture())).thenReturn(Mono.empty());

        ServerWebExchange exchange = exchange(WS_PATH + "?token=jwt-1", true);
        new UserAuthFilter(validator, properties(), new ObjectMapper(), internalTokenUtil)
                .filter(exchange, chain).block();

        ServerHttpRequest forwarded = captor.getValue().getRequest();
        assertTrue(forwarded.getURI().getRawQuery() == null
                || !forwarded.getURI().getRawQuery().contains("token="));
        assertEquals("internal-token", forwarded.getHeaders().getFirst("X-Internal-Token"));
        assertEquals("Bearer jwt-1", forwarded.getHeaders().getFirst("X-Original-Token"));
        assertNull(forwarded.getHeaders().getFirst("Authorization"));
        assertEquals("7", forwarded.getHeaders().getFirst(RequestHeaderKeys.USER_ID.getValue()));
        assertEquals("ADMIN", forwarded.getHeaders().getFirst(RequestHeaderKeys.USER_ROLE.getValue()));
    }

    @Test
    void rejectsWsHandshakeWithInvalidToken() {
        UserTokenValidator validator = Mockito.mock(UserTokenValidator.class);
        InternalTokenUtil internalTokenUtil = Mockito.mock(InternalTokenUtil.class);
        GatewayFilterChain chain = Mockito.mock(GatewayFilterChain.class);

        ServerWebExchange exchange = exchange(WS_PATH + "?token=bad", true);
        new UserAuthFilter(validator, properties(), new ObjectMapper(), internalTokenUtil)
                .filter(exchange, chain).block();

        assertEquals(HttpStatus.UNAUTHORIZED, exchange.getResponse().getStatusCode());
        Mockito.verify(chain, Mockito.never()).filter(Mockito.any(ServerWebExchange.class));
    }

    @Test
    void rejectsWsHandshakeWithoutToken() {
        UserTokenValidator validator = Mockito.mock(UserTokenValidator.class);
        InternalTokenUtil internalTokenUtil = Mockito.mock(InternalTokenUtil.class);

        ServerWebExchange exchange = exchange(WS_PATH, true);
        new UserAuthFilter(validator, properties(), new ObjectMapper(), internalTokenUtil)
                .filter(exchange, Mockito.mock(GatewayFilterChain.class)).block();

        assertEquals(HttpStatus.UNAUTHORIZED, exchange.getResponse().getStatusCode());
        Mockito.verifyNoInteractions(validator);
    }

    @Test
    void queryTokenIsAcceptedOnlyForWebSocketUpgrades() {
        UserTokenValidator validator = Mockito.mock(UserTokenValidator.class);
        InternalTokenUtil internalTokenUtil = Mockito.mock(InternalTokenUtil.class);

        ServerWebExchange exchange = exchange("/factory-service/api/digital-twin/devices?token=jwt-1", false);
        new UserAuthFilter(validator, properties(), new ObjectMapper(), internalTokenUtil)
                .filter(exchange, Mockito.mock(GatewayFilterChain.class)).block();

        assertEquals(HttpStatus.UNAUTHORIZED, exchange.getResponse().getStatusCode());
        Mockito.verifyNoInteractions(validator);
    }

    private ServerWebExchange exchange(String path, boolean websocket) {
        MockServerHttpRequest.BaseBuilder<?> builder = MockServerHttpRequest.get(path);
        if (websocket) {
            builder = builder.header("Upgrade", "websocket");
        }
        return MockServerWebExchange.from(builder.build());
    }

    private GatewayAuthProperties properties() {
        return new GatewayAuthProperties(
                "Authorization",
                "Bearer ",
                "/",
                List.of("/mall-user-service/admin/login"),
                List.of("/v3/api-docs", "/swagger-ui"),
                List.of(),
                Map.of(),
                Map.of());
    }
}
