package com.metawebthree.gateway.graphql;

import com.apollographql.federation.graphqljava.Federation;
import graphql.GraphQL;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class FederationRouter {

    private static final Logger log = LoggerFactory.getLogger(FederationRouter.class);

    private final RestTemplate restTemplate;

    private GraphQL graphQL;

    static final Map<String, String> SUBGRAPH_URLS = new LinkedHashMap<>();
    static {
        SUBGRAPH_URLS.put("product", "http://product-service/graphql");
        SUBGRAPH_URLS.put("order", "http://order-service/graphql");
        SUBGRAPH_URLS.put("user", "http://user-service/graphql");
        SUBGRAPH_URLS.put("inventory", "http://inventory-service/graphql");
        SUBGRAPH_URLS.put("recommendation", "http://recommendation-service/graphql");
        SUBGRAPH_URLS.put("cart", "http://cart-service/graphql");
    }

    static final Map<String, String> ROOT_FIELD_OWNER = new HashMap<>();
    static {
        ROOT_FIELD_OWNER.put("product", "product");
        ROOT_FIELD_OWNER.put("products", "product");
        ROOT_FIELD_OWNER.put("productBySku", "product");
        ROOT_FIELD_OWNER.put("category", "product");
        ROOT_FIELD_OWNER.put("categories", "product");
        ROOT_FIELD_OWNER.put("createProduct", "product");
        ROOT_FIELD_OWNER.put("updateProduct", "product");
        ROOT_FIELD_OWNER.put("deleteProduct", "product");

        ROOT_FIELD_OWNER.put("order", "order");
        ROOT_FIELD_OWNER.put("orders", "order");
        ROOT_FIELD_OWNER.put("orderByOrderNo", "order");
        ROOT_FIELD_OWNER.put("createOrder", "order");
        ROOT_FIELD_OWNER.put("cancelOrder", "order");
        ROOT_FIELD_OWNER.put("payOrder", "order");

        ROOT_FIELD_OWNER.put("user", "user");
        ROOT_FIELD_OWNER.put("users", "user");

        ROOT_FIELD_OWNER.put("inventory", "inventory");
        ROOT_FIELD_OWNER.put("inventoryAlerts", "inventory");

        ROOT_FIELD_OWNER.put("recommendations", "recommendation");
        ROOT_FIELD_OWNER.put("recommendationsByScene", "recommendation");
        ROOT_FIELD_OWNER.put("recommendationsByAlgorithm", "recommendation");
        ROOT_FIELD_OWNER.put("recommendation", "recommendation");
        ROOT_FIELD_OWNER.put("recommendationMetrics", "recommendation");
        ROOT_FIELD_OWNER.put("userBehaviorHistory", "recommendation");
        ROOT_FIELD_OWNER.put("rulesByScene", "recommendation");
        ROOT_FIELD_OWNER.put("generateRecommendation", "recommendation");
        ROOT_FIELD_OWNER.put("recordBehavior", "recommendation");
        ROOT_FIELD_OWNER.put("createRecommendationRule", "recommendation");
        ROOT_FIELD_OWNER.put("activateRecommendationRule", "recommendation");
        ROOT_FIELD_OWNER.put("deleteRecommendationRule", "recommendation");
        ROOT_FIELD_OWNER.put("markRecommendationClicked", "recommendation");
        ROOT_FIELD_OWNER.put("markRecommendationPurchased", "recommendation");

        ROOT_FIELD_OWNER.put("cart", "cart");
        ROOT_FIELD_OWNER.put("addToCart", "cart");
        ROOT_FIELD_OWNER.put("removeFromCart", "cart");
        ROOT_FIELD_OWNER.put("clearCart", "cart");
    }

    public FederationRouter(@LoadBalanced RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @PostConstruct
    public void init() {
        log.info("FederationRouter initializing...");
        String sdl = buildCombinedSdl();
        TypeDefinitionRegistry registry = new SchemaParser().parse(sdl);
        RuntimeWiring wiring = RuntimeWiring.newRuntimeWiring().build();
        GraphQLSchema schema = Federation.transform(registry, wiring)
                .fetchEntities(env -> {
                    List<Map<String, Object>> representations = env.getArgument("representations");
                    if (representations == null || representations.isEmpty()) return null;
                    Map<String, Object> rep = representations.get(0);
                    String typeName = (String) rep.get("__typename");
                    String owner = null;
                    for (Map.Entry<String, String> e : ENTITY_TYPES.entrySet()) {
                        if (e.getValue().equals(typeName)) { owner = e.getKey(); break; }
                    }
                    if (owner == null) return null;
                    String url = SUBGRAPH_URLS.get(owner);
                    Map<String, Object> vars = new HashMap<>();
                    vars.put("representations", representations);
                    Map<String, Object> result = executeOnSubgraph(url,
                        "query($representations: [_Any!]!) { _entities(representations: $representations) { __typename } }",
                        vars);
                    if (result != null && result.containsKey("data")) {
                        Map<String, Object> data = (Map<String, Object>) result.get("data");
                        if (data != null && data.containsKey("_entities")) {
                            List<Object> entities = (List<Object>) data.get("_entities");
                            return entities != null && !entities.isEmpty() ? entities.get(0) : null;
                        }
                    }
                    return null;
                })
                .resolveEntityType(env -> {
                    Object src = env.getObject();
                    if (src instanceof Map) {
                        String tn = (String) ((Map<?, ?>) src).get("__typename");
                        return env.getSchema().getObjectType(tn);
                    }
                    return null;
                })
                .build();
        this.graphQL = GraphQL.newGraphQL(schema).build();
        log.info("FederationRouter initialized with supergraph schema");
    }

    public GraphQL getGraphQL() {
        return graphQL;
    }

    public String resolveSubgraphUrl(String query) {
        String rootField = extractFirstRootField(query);
        if (rootField == null) return null;
        String owner = ROOT_FIELD_OWNER.get(rootField);
        if (owner == null) return null;
        return SUBGRAPH_URLS.get(owner);
    }

    static String extractFirstRootField(String query) {
        query = query.trim();
        int braceIdx = query.indexOf('{');
        if (braceIdx < 0) return null;
        String afterOperation = query.substring(braceIdx + 1).trim();
        if (afterOperation.isEmpty()) return null;
        String rest = afterOperation.split("\\{")[0].trim();
        String firstField = rest.split("[\\s(]")[0].trim();
        return firstField.isEmpty() ? null : firstField;
    }

    public Map<String, Object> executeOnSubgraph(String url, String query, Map<String, Object> variables) {
        Map<String, Object> body = new HashMap<>();
        body.put("query", query);
        if (variables != null && !variables.isEmpty()) {
            body.put("variables", variables);
        }
        try {
            return restTemplate.postForObject(url, body, Map.class);
        } catch (Exception e) {
            log.error("Subgraph call failed: url={}, error={}", url, e.getMessage());
            Map<String, Object> errorMap = new HashMap<>();
            Map<String, Object> err = new HashMap<>();
            err.put("message", "Subgraph error: " + e.getMessage());
            errorMap.put("errors", List.of(err));
            return errorMap;
        }
    }

    static final Map<String, String> ENTITY_TYPES = new HashMap<>();
    static {
        ENTITY_TYPES.put("product", "Product");
        ENTITY_TYPES.put("product", "Category");
        ENTITY_TYPES.put("order", "Order");
        ENTITY_TYPES.put("order", "OrderItem");
        ENTITY_TYPES.put("user", "User");
        ENTITY_TYPES.put("inventory", "Inventory");
        ENTITY_TYPES.put("inventory", "InventoryAlert");
        ENTITY_TYPES.put("recommendation", "Recommendation");
        ENTITY_TYPES.put("recommendation", "RecommendationResult");
        ENTITY_TYPES.put("recommendation", "RecommendationRule");
        ENTITY_TYPES.put("recommendation", "UserBehavior");
    }

    private String buildCombinedSdl() {
        StringBuilder sb = new StringBuilder();
        sb.append("type Query\n");
        sb.append("type Mutation\n");
        sb.append("type Product @key(fields: \"id\") { id: ID! name: String! }\n");
        sb.append("type Category @key(fields: \"id\") { id: ID! name: String! }\n");
        sb.append("type Order @key(fields: \"id\") { id: ID! orderNo: String! userId: ID! totalAmount: Float! }\n");
        sb.append("type OrderItem @key(fields: \"id\") { id: ID! productId: ID! }\n");
        sb.append("type User @key(fields: \"id\") { id: ID! username: String! }\n");
        sb.append("type Inventory @key(fields: \"productId\") { productId: ID! quantity: Int! }\n");
        sb.append("type InventoryAlert @key(fields: \"id\") { id: ID! productId: ID! }\n");
        sb.append("type Recommendation @key(fields: \"id\") { id: ID! userId: ID! }\n");
        sb.append("type RecommendationResult @key(fields: \"id\") { id: ID! userId: ID! productId: ID! }\n");
        sb.append("type RecommendationRule @key(fields: \"id\") { id: ID! ruleName: String! }\n");
        sb.append("type UserBehavior @key(fields: \"id\") { id: ID! userId: ID! productId: ID! }\n");
        sb.append("type ProductConnection { edges: [ProductEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type ProductEdge { node: Product cursor: String }\n");
        sb.append("type OrderConnection { edges: [OrderEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type OrderEdge { node: Order cursor: String }\n");
        sb.append("type UserConnection { edges: [UserEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type UserEdge { node: User cursor: String }\n");
        sb.append("type CategoryConnection { edges: [CategoryEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type CategoryEdge { node: Category cursor: String }\n");
        sb.append("type PageInfo { hasNextPage: Boolean! hasPreviousPage: Boolean! startCursor: String endCursor: String }\n");
        sb.append("type RecommendedItem { skuCode: String! skuName: String score: Float rank: Int reason: String }\n");
        sb.append("type RecommendationMetrics { totalRecommendations: Long! clickedCount: Long! purchasedCount: Long! clickThroughRate: Float! conversionRate: Float! }\n");
        sb.append("input CreateProductInput { name: String! sku: String! price: Float! stock: Int! categoryId: ID description: String images: [String] }\n");
        sb.append("input UpdateProductInput { name: String price: Float stock: Int categoryId: ID description: String images: [String] status: String }\n");
        sb.append("input CreateOrderInput { items: [OrderItemInput!]! shippingAddress: String! paymentMethod: String! }\n");
        sb.append("input OrderItemInput { productId: ID! quantity: Int! }\n");
        return sb.toString();
    }
}
