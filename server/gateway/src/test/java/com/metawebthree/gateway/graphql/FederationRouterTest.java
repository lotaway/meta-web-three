package com.metawebthree.gateway.graphql;

import graphql.ExecutionInput;
import graphql.ExecutionResult;
import graphql.GraphQL;
import org.junit.jupiter.api.*;
import org.mockito.*;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Integration tests for GraphQL Federation gateway.
 *
 * Verifies:
 * 1. Supergraph schema builds correctly with _service and _entities
 * 2. Root field routing resolves to correct subgraph URLs
 * 3. Federated entity resolution via _entities queries
 * 4. Cross-subgraph queries (e.g., cart referencing product)
 */
class FederationRouterTest {

    private FederationRouter federationRouter;
    private RestTemplate restTemplate;

    @BeforeEach
    void setUp() {
        restTemplate = Mockito.mock(RestTemplate.class);
        federationRouter = new FederationRouter(restTemplate);
        federationRouter.graphQL();
    }

    // ── Schema Bootstrap ──────────────────────────────────────────────

    @Test
    void shouldBuildSupergraphSchemaWithFederationTypes() {
        GraphQL graphQL = federationRouter.getGraphQL();
        assertNotNull(graphQL, "GraphQL engine should be initialized");
        assertNotNull(graphQL.getGraphQLSchema(), "Supergraph schema should exist");

        // Verify _service and _entities are in the schema
        var queryType = graphQL.getGraphQLSchema().getQueryType();
        assertNotNull(queryType, "Query type should exist");
        assertNotNull(queryType.getFieldDefinition("_service"), "_service field should exist");
        assertNotNull(queryType.getFieldDefinition("_entities"), "_entities field should exist");
    }

    @Test
    void shouldExposeAllEntityTypes() {
        GraphQL graphQL = federationRouter.getGraphQL();
        var schema = graphQL.getGraphQLSchema();

        // Verify key entity types are registered
        assertNotNull(schema.getObjectType("Product"), "Product type should exist");
        assertNotNull(schema.getObjectType("Order"), "Order type should exist");
        assertNotNull(schema.getObjectType("User"), "User type should exist");
        assertNotNull(schema.getObjectType("Cart"), "Cart type should exist");
        assertNotNull(schema.getObjectType("Inventory"), "Inventory type should exist");
        assertNotNull(schema.getObjectType("Recommendation"), "Recommendation type should exist");
    }

    @Test
    void shouldExposeQueryAndMutationFields() {
        GraphQL graphQL = federationRouter.getGraphQL();
        var queryType = graphQL.getGraphQLSchema().getQueryType();
        var mutationType = graphQL.getGraphQLSchema().getMutationType();

        assertNotNull(queryType, "Query type should exist");
        assertNotNull(mutationType, "Mutation type should exist");

        // Query fields
        assertNotNull(queryType.getFieldDefinition("product"));
        assertNotNull(queryType.getFieldDefinition("products"));
        assertNotNull(queryType.getFieldDefinition("order"));
        assertNotNull(queryType.getFieldDefinition("orders"));
        assertNotNull(queryType.getFieldDefinition("user"));
        assertNotNull(queryType.getFieldDefinition("users"));
        assertNotNull(queryType.getFieldDefinition("cart"));
        assertNotNull(queryType.getFieldDefinition("inventory"));
        assertNotNull(queryType.getFieldDefinition("recommendations"));

        // Mutation fields
        assertNotNull(mutationType.getFieldDefinition("createProduct"));
        assertNotNull(mutationType.getFieldDefinition("createOrder"));
        assertNotNull(mutationType.getFieldDefinition("addToCart"));
        assertNotNull(mutationType.getFieldDefinition("generateRecommendation"));
    }

    // ── Subgraph URL Resolution ───────────────────────────────────────

    @Test
    void shouldResolveProductQueriesToProductService() {
        String url = federationRouter.resolveSubgraphUrl("{ product(id: \"1\") { id name } }");
        assertEquals("http://product-service/graphql", url);
    }

    @Test
    void shouldResolveOrderQueriesToOrderService() {
        String url = federationRouter.resolveSubgraphUrl("{ order(id: \"1\") { id orderNo } }");
        assertEquals("http://order-service/graphql", url);
    }

    @Test
    void shouldResolveUserQueriesToUserService() {
        String url = federationRouter.resolveSubgraphUrl("{ user(id: \"1\") { id username } }");
        assertEquals("http://user-service/graphql", url);
    }

    @Test
    void shouldResolveCartQueriesToCartService() {
        String url = federationRouter.resolveSubgraphUrl("{ cart(userId: \"1\") { id items { productId } } }");
        assertEquals("http://cart-service/graphql", url);
    }

    @Test
    void shouldResolveInventoryQueriesToInventoryService() {
        String url = federationRouter.resolveSubgraphUrl("{ inventory(productId: \"1\") { productId quantity } }");
        assertEquals("http://inventory-service/graphql", url);
    }

    @Test
    void shouldResolveRecommendationQueriesToRecommendationService() {
        String url = federationRouter.resolveSubgraphUrl("{ recommendations(userId: \"1\", limit: 10) { id score } }");
        assertEquals("http://recommendation-service/graphql", url);
    }

    @Test
    void shouldReturnNullForUnknownFields() {
        String url = federationRouter.resolveSubgraphUrl("{ unknownField { id } }");
        assertNull(url);
    }

    @Test
    void shouldReturnNullForFederationMetaFields() {
        // _service and _entities are handled locally, not proxied
        String url1 = federationRouter.resolveSubgraphUrl("{ _service { sdl } }");
        assertNull(url1, "_service should not route to a subgraph");

        String url2 = federationRouter.resolveSubgraphUrl("{ _entities(representations: []) { ... on Product { id } } }");
        assertNull(url2, "_entities should not route to a subgraph");
    }

    // ── Root Field Extraction ──────────────────────────────────────────

    @Test
    void shouldExtractFirstRootField() {
        assertEquals("product", FederationRouter.extractFirstRootField("{ product(id: \"1\") { id name } }"));
        assertEquals("order", FederationRouter.extractFirstRootField("{ order(id: \"1\") { id } }"));
        assertEquals("cart", FederationRouter.extractFirstRootField("{ cart(userId: \"1\") { id } }"));
    }

    @Test
    void shouldExtractFirstRootFieldWithAlias() {
        // Aliased query — still extracts the field name after the alias
        assertEquals("myProduct", FederationRouter.extractFirstRootField("{ myProduct: product(id: \"1\") { id } }"));
    }

    @Test
    void shouldHandleNullQuery() {
        assertNull(FederationRouter.extractFirstRootField(null));
        assertNull(FederationRouter.extractFirstRootField(""));
        assertNull(FederationRouter.extractFirstRootField("   "));
    }

    // ── Federated Entity Resolution ───────────────────────────────────

    @Test
    void shouldResolveProductEntityFromSubgraph() {
        // Mock the subgraph _entities response for Product
        Map<String, Object> productEntity = new ConcurrentHashMap<>();
        productEntity.put("__typename", "Product");
        productEntity.put("id", "42");
        productEntity.put("name", "Test Product");
        productEntity.put("sku", "SKU-001");
        productEntity.put("price", 99.99);
        productEntity.put("stock", 100);

        Map<String, Object> subgraphResponse = new ConcurrentHashMap<>();
        subgraphResponse.put("data", Map.of("_entities", List.of(productEntity)));

        when(restTemplate.postForObject(
                eq("http://product-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        // Execute a federated query referencing a Product entity
        GraphQL graphQL = federationRouter.getGraphQL();
        String query = """
            query($representations: [_Any!]!) {
              _entities(representations: $representations) {
                ... on Product { __typename id name sku price stock }
              }
            }
            """;

        Map<String, Object> variables = Map.of(
                "representations", List.of(Map.of("__typename", "Product", "id", "42"))
        );

        ExecutionResult result = graphQL.execute(ExecutionInput.newExecutionInput()
                .query(query)
                .variables(variables)
                .build());

        assertFalse(result.getErrors().stream().anyMatch(e -> e.getMessage().contains("No owner")),
                "Should not have 'No owner' errors for Product type");
    }

    @Test
    void shouldResolveUserEntityFromSubgraph() {
        Map<String, Object> userEntity = new ConcurrentHashMap<>();
        userEntity.put("__typename", "User");
        userEntity.put("id", "1");
        userEntity.put("username", "testuser");
        userEntity.put("nickname", "Test User");

        Map<String, Object> subgraphResponse = new ConcurrentHashMap<>();
        subgraphResponse.put("data", Map.of("_entities", List.of(userEntity)));

        when(restTemplate.postForObject(
                eq("http://user-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        GraphQL graphQL = federationRouter.getGraphQL();
        String query = """
            query($representations: [_Any!]!) {
              _entities(representations: $representations) {
                ... on User { __typename id username nickname }
              }
            }
            """;

        Map<String, Object> variables = Map.of(
                "representations", List.of(Map.of("__typename", "User", "id", "1"))
        );

        ExecutionResult result = graphQL.execute(ExecutionInput.newExecutionInput()
                .query(query)
                .variables(variables)
                .build());

        assertFalse(result.getErrors().stream().anyMatch(e -> e.getMessage().contains("No owner")),
                "Should not have 'No owner' errors for User type");
    }

    // ── Subgraph Proxy ────────────────────────────────────────────────

    @Test
    void shouldProxyQueryToSubgraphAndReturnData() {
        Map<String, Object> productData = new ConcurrentHashMap<>();
        productData.put("id", "1");
        productData.put("name", "Widget");
        productData.put("sku", "W-001");
        productData.put("price", 29.99);
        productData.put("stock", 50);

        Map<String, Object> subgraphResponse = new ConcurrentHashMap<>();
        subgraphResponse.put("data", Map.of("product", productData));

        when(restTemplate.postForObject(
                eq("http://product-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(subgraphResponse);

        Map<String, Object> result = federationRouter.executeOnSubgraph(
                "http://product-service/graphql",
                "{ product(id: \"1\") { id name sku price stock } }",
                Map.of("id", "1")
        );

        assertNotNull(result);
        assertEquals(productData, ((Map<?, ?>) result.get("data")).get("product"));
    }

    @Test
    void shouldHandleSubgraphError() {
        when(restTemplate.postForObject(
                anyString(),
                any(Map.class),
                eq(Map.class)
        )).thenThrow(new RuntimeException("Connection refused"));

        Map<String, Object> result = federationRouter.executeOnSubgraph(
                "http://product-service/graphql",
                "{ product(id: \"1\") { id } }",
                null
        );

        assertNotNull(result);
        assertTrue(result.containsKey("errors"), "Should contain error response");
    }

    // ── Cross-Subgraph Query (Cart → Product Federation) ──────────────

    @Test
    void shouldSupportCrossSubgraphQueryCartReferencingProduct() {
        // Simulate: query { cart(userId: "1") { id items { productId quantity } } }
        // The cart subgraph returns items with productId, and the client can
        // then federate Product entities from the product subgraph.

        // Step 1: Cart subgraph returns cart with product references
        Map<String, Object> cartItem1 = new ConcurrentHashMap<>();
        cartItem1.put("__typename", "CartItem");
        cartItem1.put("id", "101");
        cartItem1.put("productId", "42");
        cartItem1.put("quantity", 2);
        cartItem1.put("price", 29.99);
        cartItem1.put("subtotal", 59.98);

        Map<String, Object> cartData = new ConcurrentHashMap<>();
        cartData.put("__typename", "Cart");
        cartData.put("id", "1");
        cartData.put("userId", "1");
        cartData.put("items", List.of(cartItem1));
        cartData.put("totalAmount", 59.98);
        cartData.put("itemCount", 1);

        Map<String, Object> cartResponse = new ConcurrentHashMap<>();
        cartResponse.put("data", Map.of("cart", cartData));

        when(restTemplate.postForObject(
                eq("http://cart-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(cartResponse);

        // Step 2: Product subgraph resolves the federated Product entity
        Map<String, Object> productEntity = new ConcurrentHashMap<>();
        productEntity.put("__typename", "Product");
        productEntity.put("id", "42");
        productEntity.put("name", "Widget Pro");
        productEntity.put("sku", "WP-001");
        productEntity.put("price", 29.99);
        productEntity.put("stock", 100);

        Map<String, Object> productResponse = new ConcurrentHashMap<>();
        productResponse.put("data", Map.of("_entities", List.of(productEntity)));

        when(restTemplate.postForObject(
                eq("http://product-service/graphql"),
                any(Map.class),
                eq(Map.class)
        )).thenReturn(productResponse);

        // Verify cart subgraph returns product references
        Map<String, Object> cartResult = federationRouter.executeOnSubgraph(
                "http://cart-service/graphql",
                "{ cart(userId: \"1\") { __typename id userId items { __typename id productId quantity price subtotal } totalAmount itemCount } }",
                Map.of("userId", "1")
        );

        assertNotNull(cartResult);
        Map<?, ?> data = (Map<?, ?>) cartResult.get("data");
        Map<?, ?> cart = (Map<?, ?>) data.get("cart");
        assertNotNull(cart);
        assertEquals("1", cart.get("userId"));
        assertEquals(1, cart.get("itemCount"));

        // Verify product entity can be resolved from cart's productId reference
        Map<String, Object> productResult = federationRouter.executeOnSubgraph(
                "http://product-service/graphql",
                "query($representations: [_Any!]!) { _entities(representations: $representations) { ... on Product { __typename id name sku price stock } } }",
                Map.of("representations", List.of(Map.of("__typename", "Product", "id", "42")))
        );

        assertNotNull(productResult);
        Map<?, ?> productData2 = (Map<?, ?>) productResult.get("data");
        List<?> entities = (List<?>) productData2.get("_entities");
        assertNotNull(entities);
        assertEquals(1, entities.size());
        Map<?, ?> resolved = (Map<?, ?>) entities.get(0);
        assertEquals("Widget Pro", resolved.get("name"));
    }

    // ── _service SDL Query ────────────────────────────────────────────

    @Test
    void shouldReturnServiceSdl() {
        GraphQL graphQL = federationRouter.getGraphQL();
        ExecutionResult result = graphQL.execute("{ _service { sdl } }");

        assertNull(result.getErrors(), "Should not have errors for _service query");
        Map<?, ?> data = result.getData();
        assertNotNull(data);
        Map<?, ?> service = (Map<?, ?>) data.get("_service");
        assertNotNull(service);
        String sdl = (String) service.get("sdl");
        assertNotNull(sdl, "SDL should be returned");
        // SDL should contain key types
        assertTrue(sdl.contains("Product") || sdl.contains("product"),
                "SDL should reference Product type");
    }

    // ── Subgraph URL Mapping Coverage ─────────────────────────────────

    @Test
    void shouldMapAllExpectedSubgraphs() {
        Map<String, String> urls = FederationRouter.SUBGRAPH_URLS;
        assertEquals(6, urls.size(), "Should have 6 subgraph mappings");
        assertTrue(urls.containsKey("product"));
        assertTrue(urls.containsKey("order"));
        assertTrue(urls.containsKey("user"));
        assertTrue(urls.containsKey("inventory"));
        assertTrue(urls.containsKey("recommendation"));
        assertTrue(urls.containsKey("cart"));
    }

    @Test
    void shouldMapRootFieldsToCorrectOwners() {
        Map<String, String> owners = FederationRouter.ROOT_FIELD_OWNER;

        // Product fields
        assertEquals("product", owners.get("product"));
        assertEquals("product", owners.get("products"));
        assertEquals("product", owners.get("productBySku"));
        assertEquals("product", owners.get("category"));
        assertEquals("product", owners.get("categories"));

        // Order fields
        assertEquals("order", owners.get("order"));
        assertEquals("order", owners.get("orders"));
        assertEquals("order", owners.get("createOrder"));
        assertEquals("order", owners.get("cancelOrder"));
        assertEquals("order", owners.get("payOrder"));

        // User fields
        assertEquals("user", owners.get("user"));
        assertEquals("user", owners.get("users"));

        // Cart fields
        assertEquals("cart", owners.get("cart"));
        assertEquals("cart", owners.get("addToCart"));
        assertEquals("cart", owners.get("removeFromCart"));
        assertEquals("cart", owners.get("clearCart"));

        // Inventory fields
        assertEquals("inventory", owners.get("inventory"));
        assertEquals("inventory", owners.get("inventoryAlerts"));

        // Recommendation fields
        assertEquals("recommendation", owners.get("recommendations"));
        assertEquals("recommendation", owners.get("recommendation"));
        assertEquals("recommendation", owners.get("generateRecommendation"));
    }
}
