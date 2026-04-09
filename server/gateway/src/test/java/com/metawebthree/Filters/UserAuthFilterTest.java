package com.metawebthree.Filters;

import org.junit.jupiter.api.Assertions;
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
import com.metawebthree.gateway.auth.GatewayAuthProperties;
import com.metawebthree.gateway.auth.UserTokenClaims;
import com.metawebthree.gateway.auth.UserTokenValidator;

import reactor.core.publisher.Mono;

public class UserAuthFilterTest {

    @Test
    public void shouldSkipAuthenticationWhenPathOutsideProtectedPrefix() {
        UserTokenValidator tokenValidator = Mockito.mock(UserTokenValidator.class);
        UserAuthFilter filter = new UserAuthFilter(tokenValidator, defaultProperties(), new ObjectMapper());
        GatewayFilterChain chain = Mockito.mock(GatewayFilterChain.class);
        ServerWebExchange exchange = MockServerWebExchange.from(MockServerHttpRequest.get("/product-service/product/list").build());
        Mockito.when(chain.filter(exchange)).thenReturn(Mono.empty());

        filter.filter(exchange, chain).block();

        Mockito.verify(chain).filter(exchange);
        Mockito.verifyNoInteractions(tokenValidator);
    }

    @Test
    public void shouldReturnUnauthorizedWhenAuthorizationHeaderMissing() {
        UserTokenValidator tokenValidator = Mockito.mock(UserTokenValidator.class);
        UserAuthFilter filter = new UserAuthFilter(tokenValidator, defaultProperties(), new ObjectMapper());
        ServerWebExchange exchange = MockServerWebExchange.from(MockServerHttpRequest.get("/user-service/user/profile").build());

        filter.filter(exchange, Mockito.mock(GatewayFilterChain.class)).block();

        Assertions.assertEquals(HttpStatus.UNAUTHORIZED, exchange.getResponse().getStatusCode());
        String body = ((MockServerWebExchange) exchange).getResponse().getBodyAsString().block();
        Assertions.assertTrue(body.contains("AUTH_HEADER_INVALID"));
    }

    @Test
    public void shouldInjectUserHeadersWhenTokenIsValid() {
        UserTokenValidator tokenValidator = Mockito.mock(UserTokenValidator.class);
        UserAuthFilter filter = new UserAuthFilter(tokenValidator, defaultProperties(), new ObjectMapper());
        GatewayFilterChain chain = Mockito.mock(GatewayFilterChain.class);
        ArgumentCaptor<ServerWebExchange> exchangeCaptor = ArgumentCaptor.forClass(ServerWebExchange.class);
        Mockito.when(chain.filter(exchangeCaptor.capture())).thenReturn(Mono.empty());
        Mockito.when(tokenValidator.validate("valid-token"))
                .thenReturn(new UserTokenClaims(7L, "alice", "USER"));
        ServerWebExchange exchange = MockServerWebExchange.from(
                MockServerHttpRequest.get("/user-service/user/profile")
                        .header("Authorization", "Bearer valid-token")
                        .build());

        filter.filter(exchange, chain).block();

        ServerHttpRequest forwardedRequest = exchangeCaptor.getValue().getRequest();
        Assertions.assertEquals("7", forwardedRequest.getHeaders().getFirst(RequestHeaderKeys.USER_ID.getValue()));
        Assertions.assertEquals("alice", forwardedRequest.getHeaders().getFirst(RequestHeaderKeys.USER_NAME.getValue()));
        Assertions.assertEquals("USER", forwardedRequest.getHeaders().getFirst(RequestHeaderKeys.USER_ROLE.getValue()));
    }

    private GatewayAuthProperties defaultProperties() {
        return new GatewayAuthProperties(
                "Authorization",
                "Bearer ",
                "/user-service/",
                java.util.List.of(
                        "/user-service/user/signIn",
                        "/user-service/user/create",
                        "/user-service/user/checkWeb3SignerMessage",
                        "/actuator"),
                java.util.List.of("/v3/api-docs", "/swagger-ui"));
    }
}
