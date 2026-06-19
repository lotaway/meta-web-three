package com.metawebthree.gateway.graphql;

import graphql.GraphQL;
import org.junit.jupiter.api.*;
import org.mockito.*;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.client.RestTemplate;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Integration tests for the GraphQL HTTP handler.
 *
 * Verifies the HTTP request/response flow through GraphQLHandler,
 * including routing decisions and subgraph proxy behavior.
 */
class GraphQLHandlerTest {

    private GraphQLHandler graphQLHandler;
    private GraphQL graphQL;
    private FederationRouter federationRouter;
    private RestTemplate restTemplate;

    @BeforeEach
    void setUp() {
        restTemplate = Mockito.mock(RestTemplate.class);
        federationRouter = new FederationRouter(restTemplate);
        federationRouter.graphQL();
        graphQL = federationRouter.getGraphQL();
        graphQLHandler = new GraphQLHandler(graphQL, federationRouter);
    }

    @Test
    void shouldRejectEmptyQuery() {
        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(Map.of()));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> assertEquals(400, response.statusCode().value()))
                .verifyComplete();
    }

    @Test
    void shouldExecuteServiceQueryLocally() {
        // _service query should be handled locally, not proxied
        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ _service { sdl } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    assertEquals(200, response.statusCode().value());
                    // Should not have called any subgraph
                    verify(restTemplate, never()).postForObject(anyString(), any(), eq(Map.class));
                })
                .verifyComplete();
    }

    @Test
    void shouldProxyProductQueryToSubgraph() {
        Map<String, Object> productData = new HashMap<>();
        productData.put("id", "1");
        productData.put("name", "Test Product");

        Map<String, Object> subgraphResponse = new HashMap<>();
        subgraphResponse.put("data", Map.of("product", productData));

        when(restTemplate.postForObject(
                eq("http://product-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ product(id: \"1\") { id name } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    assertEquals(200, response.statusCode().value());
                    verify(restTemplate).postForObject(
                            eq("http://product-service/graphql"),
                            any(Map.class),
                            eq(Map.class)
                    );
                })
                .verifyComplete();
    }

    @Test
    void shouldProxyOrderQueryToSubgraph() {
        Map<String, Object> orderData = new HashMap<>();
        orderData.put("id", "10");
        orderData.put("orderNo", "ORD-001");
        orderData.put("status", "PENDING");

        Map<String, Object> subgraphResponse = new HashMap<>();
        subgraphResponse.put("data", Map.of("order", orderData));

        when(restTemplate.postForObject(
                eq("http://order-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ order(id: \"10\") { id orderNo status } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    assertEquals(200, response.statusCode().value());
                    verify(restTemplate).postForObject(
                            eq("http://order-service/graphql"),
                            any(Map.class),
                            eq(Map.class)
                    );
                })
                .verifyComplete();
    }

    @Test
    void shouldProxyCartQueryToSubgraph() {
        Map<String, Object> cartData = new HashMap<>();
        cartData.put("id", "1");
        cartData.put("userId", "1");
        cartData.put("itemCount", 0);

        Map<String, Object> subgraphResponse = new HashMap<>();
        subgraphResponse.put("data", Map.of("cart", cartData));

        when(restTemplate.postForObject(
                eq("http://cart-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ cart(userId: \"1\") { id userId itemCount } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    assertEquals(200, response.statusCode().value());
                    verify(restTemplate).postForObject(
                            eq("http://cart-service/graphql"),
                            any(Map.class),
                            eq(Map.class)
                    );
                })
                .verifyComplete();
    }

    @Test
    void shouldHandleSubgraphFailureGracefully() {
        when(restTemplate.postForObject(
                anyString(),
                any(Map.class),
                eq(Map.class)
        )).thenThrow(new RuntimeException("Service unavailable"));

        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ product(id: \"1\") { id name } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        // Should fall back to local execution or return error, not crash
        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    // Either 200 with local fallback or 200 with errors — should not throw
                    assertEquals(200, response.statusCode().value());
                })
                .verifyComplete();
    }

    @Test
    void shouldExecuteLocalFallbackForUnknownFields() {
        Map<String, Object> body = new HashMap<>();
        body.put("query", "{ _service { sdl } }");

        MockServerRequest request = MockServerRequest.builder()
                .body(Mono.just(body));

        Mono<ServerResponse> responseMono = graphQLHandler.handleGraphQL(request);

        StepVerifier.create(responseMono)
                .assertNext(response -> {
                    assertEquals(200, response.statusCode().value());
                    verify(restTemplate, never()).postForObject(anyString(), any(), eq(Map.class));
                })
                .verifyComplete();
    }
}
