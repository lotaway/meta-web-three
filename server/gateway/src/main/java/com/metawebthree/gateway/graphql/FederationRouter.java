package com.metawebthree.gateway.graphql;

import com.apollographql.federation.graphqljava.Federation;
import graphql.*;
import graphql.schema.*;
import graphql.schema.idl.RuntimeWiring;
import graphql.schema.idl.SchemaParser;
import graphql.schema.idl.TypeDefinitionRegistry;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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

    static final Map<String, String> ROOT_FIELD_OWNER = new LinkedHashMap<>();
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

    static final Map<String, String> TYPE_TO_OWNER = new HashMap<>();
    static {
        TYPE_TO_OWNER.put("Product", "product");
        TYPE_TO_OWNER.put("Category", "product");
        TYPE_TO_OWNER.put("Order", "order");
        TYPE_TO_OWNER.put("OrderItem", "order");
        TYPE_TO_OWNER.put("User", "user");
        TYPE_TO_OWNER.put("Inventory", "inventory");
        TYPE_TO_OWNER.put("InventoryAlert", "inventory");
        TYPE_TO_OWNER.put("Recommendation", "recommendation");
        TYPE_TO_OWNER.put("RecommendationResult", "recommendation");
        TYPE_TO_OWNER.put("RecommendationRule", "recommendation");
        TYPE_TO_OWNER.put("UserBehavior", "recommendation");
        TYPE_TO_OWNER.put("Cart", "cart");
        TYPE_TO_OWNER.put("CartItem", "cart");
    }

    public FederationRouter(@LoadBalanced RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @PostConstruct
    public void init() {
        log.info("FederationRouter initializing...");
        String sdl = buildCombinedSdl();
        log.debug("Supergraph SDL:\n{}", sdl);
        TypeDefinitionRegistry registry = new SchemaParser().parse(sdl);
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = Federation.transform(registry, wiring)
                .fetchEntities(env -> {
                    List<Map<String, Object>> representations = env.getArgument("representations");
                    if (representations == null || representations.isEmpty())
                        return null;

                    Map<String, List<Map<String, Object>>> byType = new LinkedHashMap<>();
                    for (Map<String, Object> rep : representations) {
                        String typeName = (String) rep.get("__typename");
                        byType.computeIfAbsent(typeName, k -> new ArrayList<>()).add(rep);
                    }

                    List<Object> results = new ArrayList<>();
                    for (Map.Entry<String, List<Map<String, Object>>> entry : byType.entrySet()) {
                        String typeName = entry.getKey();
                        List<Map<String, Object>> reps = entry.getValue();
                        String owner = TYPE_TO_OWNER.get(typeName);
                        if (owner == null) {
                            log.warn("No owner found for type: {}", typeName);
                            results.addAll(reps.stream().map(r -> (Object) r).toList());
                            continue;
                        }
                        String subgraphUrl = SUBGRAPH_URLS.get(owner);
                        if (subgraphUrl == null) {
                            log.warn("No URL found for owner: {}", owner);
                            results.addAll(reps.stream().map(r -> (Object) r).toList());
                            continue;
                        }
                        String entitiesQuery = buildEntitiesQuery(typeName, reps.size());
                        Map<String, Object> vars = new ConcurrentHashMap<>();
                        vars.put("representations", reps);
                        try {
                            Map<String, Object> resp = executeOnSubgraph(subgraphUrl, entitiesQuery, vars);
                            if (resp != null && resp.containsKey("data")) {
                                @SuppressWarnings("unchecked")
                                Map<String, Object> data = (Map<String, Object>) resp.get("data");
                                if (data != null && data.containsKey("_entities")) {
                                    @SuppressWarnings("unchecked")
                                    List<Object> entities = (List<Object>) data.get("_entities");
                                    if (entities != null) {
                                        results.addAll(entities);
                                        continue;
                                    }
                                }
                            }
                            log.warn("Subgraph {} returned no entities for type {}, using raw rep", owner, typeName);
                            results.addAll(reps.stream().map(r -> (Object) r).toList());
                        } catch (Exception e) {
                            log.error("Failed to fetch entities from {} for type {}: {}", owner, typeName,
                                    e.getMessage());
                            results.addAll(reps.stream().map(r -> (Object) r).toList());
                        }
                    }
                    return results;
                })
                .resolveEntityType(env -> {
                    Object src = env.getObject();
                    if (src instanceof Map) {
                        String tn = (String) ((Map<?, ?>) src).get("__typename");
                        if (tn != null) {
                            GraphQLObjectType type = env.getSchema().getObjectType(tn);
                            if (type != null)
                                return type;
                        }
                    }
                    return null;
                })
                .build();
        this.graphQL = GraphQL.newGraphQL(schema).build();
        log.info("FederationRouter initialized with supergraph schema");
    }

    private String buildEntitiesQuery(String typeName, int count) {
        StringBuilder sb = new StringBuilder();
        sb.append("query($representations: [_Any!]!) { _entities(representations: $representations) {");
        switch (typeName) {
            case "Product":
                sb.append(
                        " ... on Product { __typename id name sku price stock categoryId description images status createdAt updatedAt }");
                break;
            case "Category":
                sb.append(" ... on Category { __typename id name parentId level sortOrder }");
                break;
            case "Order":
                sb.append(
                        " ... on Order { __typename id orderNo userId totalAmount status shippingAddress paymentMethod createdAt updatedAt }");
                break;
            case "OrderItem":
                sb.append(" ... on OrderItem { __typename id productId quantity price subtotal }");
                break;
            case "User":
                sb.append(" ... on User { __typename id username nickname email mobile avatar status createdAt }");
                break;
            case "Inventory":
                sb.append(
                        " ... on Inventory { __typename productId quantity availableQuantity reservedQuantity warehouseId updatedAt }");
                break;
            case "InventoryAlert":
                sb.append(
                        " ... on InventoryAlert { __typename id productId alertType currentStock threshold status createdAt resolvedAt }");
                break;
            case "Recommendation":
                sb.append(
                        " ... on Recommendation { __typename id userId scene algorithm score status clickCount conversionCount impressionCount createdAt expiresAt }");
                break;
            case "RecommendationResult":
                sb.append(
                        " ... on RecommendationResult { __typename id userId productId score algorithm reason position isClicked isPurchased createdAt expiresAt }");
                break;
            case "RecommendationRule":
                sb.append(
                        " ... on RecommendationRule { __typename id ruleName scene type status priority maxItems conditions exclusions boostFactor createdAt updatedAt }");
                break;
            case "UserBehavior":
                sb.append(
                        " ... on UserBehavior { __typename id userId productId behaviorType behaviorValue timestamp sessionId source }");
                break;
            case "Cart":
                sb.append(" ... on Cart { __typename id userId items totalAmount itemCount }");
                break;
            case "CartItem":
                sb.append(" ... on CartItem { __typename id productId quantity price subtotal }");
                break;
            default:
                sb.append(" __typename");
        }
        sb.append("}}");
        return sb.toString();
    }

    public GraphQL getGraphQL() {
        return graphQL;
    }

    /**
     * Resolve the owning subgraph URL for a given GraphQL query string.
     * Extracts the first root field and looks it up in ROOT_FIELD_OWNER.
     */
    public String resolveSubgraphUrl(String query) {
        String rootField = extractFirstRootField(query);
        if (rootField == null)
            return null;
        String owner = ROOT_FIELD_OWNER.get(rootField);
        if (owner == null)
            return null;
        return SUBGRAPH_URLS.get(owner);
    }

    static String extractFirstRootField(String query) {
        if (query == null)
            return null;
        query = query.trim();
        int braceIdx = query.indexOf('{');
        if (braceIdx < 0)
            return null;
        String afterOpen = query.substring(braceIdx + 1).trim();
        if (afterOpen.isEmpty())
            return null;
        // Skip fragments and directives to find the actual field name
        String firstLine = afterOpen.split("\n")[0].trim();
        // Match: fieldName(...)
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("^([a-zA-Z_][a-zA-Z0-9_]*)");
        java.util.regex.Matcher m = p.matcher(firstLine);
        if (m.find()) {
            return m.group(1);
        }
        return null;
    }

    public Map<String, Object> executeOnSubgraph(String url, String query, Map<String, Object> variables) {
        Map<String, Object> body = new ConcurrentHashMap<>();
        body.put("query", query);
        if (variables != null && !variables.isEmpty()) {
            body.put("variables", variables);
        }
        try {
            return restTemplate.postForObject(url, body, Map.class);
        } catch (Exception e) {
            log.error("Subgraph call failed: url={}, error={}", url, e.getMessage());
            Map<String, Object> errorMap = new ConcurrentHashMap<>();
            Map<String, Object> err = new ConcurrentHashMap<>();
            err.put("message", "Subgraph error: " + e.getMessage());
            errorMap.put("errors", List.of(err));
            return errorMap;
        }
    }

    /** Build RuntimeWiring with DataFetchers that proxy to subgraphs. */
    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                // ── Product ─────────────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("product", env -> proxy(env, "product", "product",
                                "{ product(id: $id) { __typename id name sku price stock categoryId description images status createdAt updatedAt } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("products", env -> proxy(env, "product", "products",
                                "{ products(page: $page, size: $size) { edges { node { __typename id name sku price stock status } } pageInfo { hasNextPage hasPreviousPage } totalCount } }",
                                Map.of("page", env.getArgument("page"), "size", env.getArgument("size"))))
                        .dataFetcher("productBySku", env -> proxy(env, "product", "productBySku",
                                "{ productBySku(sku: $sku) { __typename id name sku price stock status } }",
                                Map.of("sku", env.getArgument("sku"))))
                        .dataFetcher("category", env -> proxy(env, "product", "category",
                                "{ category(id: $id) { __typename id name parentId level sortOrder } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("categories", env -> proxy(env, "product", "categories",
                                "{ categories(parentId: $parentId) { edges { node { __typename id name parentId level sortOrder } } } }",
                                Map.of("parentId", env.getArgument("parentId")))))
                // ── Order ──────────────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("order", env -> proxy(env, "order", "order",
                                "{ order(id: $id) { __typename id orderNo userId totalAmount status shippingAddress paymentMethod createdAt updatedAt items { __typename id productId quantity price subtotal } } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("orders", env -> proxy(env, "order", "orders",
                                "{ orders(page: $page, size: $size, status: $status) { edges { node { __typename id orderNo userId totalAmount status createdAt } } pageInfo { hasNextPage } totalCount } }",
                                Map.of("page", env.getArgument("page"), "size", env.getArgument("size"), "status",
                                        env.getArgument("status"))))
                        .dataFetcher("orderByOrderNo", env -> proxy(env, "order", "orderByOrderNo",
                                "{ orderByOrderNo(orderNo: $orderNo) { __typename id orderNo userId totalAmount status } }",
                                Map.of("orderNo", env.getArgument("orderNo")))))
                // ── User ───────────────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("user", env -> proxy(env, "user", "user",
                                "{ user(id: $id) { __typename id username nickname email mobile avatar status createdAt } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("users", env -> proxy(env, "user", "users",
                                "{ users(page: $page, size: $size) { edges { node { __typename id username nickname status } } pageInfo { hasNextPage } totalCount } }",
                                Map.of("page", env.getArgument("page"), "size", env.getArgument("size")))))
                // ── Inventory ──────────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("inventory", env -> proxy(env, "inventory", "inventory",
                                "{ inventory(productId: $productId) { __typename productId quantity availableQuantity reservedQuantity warehouseId updatedAt } }",
                                Map.of("productId", env.getArgument("productId"))))
                        .dataFetcher("inventoryAlerts", env -> proxy(env, "inventory", "inventoryAlerts",
                                "{ inventoryAlerts(status: $status) { __typename id productId alertType currentStock threshold status createdAt resolvedAt } }",
                                Map.of("status", env.getArgument("status")))))
                // ── Recommendation ─────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("recommendations", env -> proxy(env, "recommendation", "recommendations",
                                "{ recommendations(userId: $userId, limit: $limit) { __typename id userId productId score algorithm reason position isClicked isPurchased } }",
                                Map.of("userId", env.getArgument("userId"), "limit", env.getArgument("limit"))))
                        .dataFetcher("recommendationsByScene", env -> proxy(env, "recommendation",
                                "recommendationsByScene",
                                "{ recommendationsByScene(userId: $userId, scene: $scene, limit: $limit) { __typename id userId productId score algorithm reason } }",
                                Map.of("userId", env.getArgument("userId"), "scene", env.getArgument("scene"), "limit",
                                        env.getArgument("limit"))))
                        .dataFetcher("recommendationsByAlgorithm", env -> proxy(env, "recommendation",
                                "recommendationsByAlgorithm",
                                "{ recommendationsByAlgorithm(userId: $userId, algorithm: $algorithm, limit: $limit) { __typename id userId productId score algorithm } }",
                                Map.of("userId", env.getArgument("userId"), "algorithm", env.getArgument("algorithm"),
                                        "limit", env.getArgument("limit"))))
                        .dataFetcher("recommendation", env -> proxy(env, "recommendation", "recommendation",
                                "{ recommendation(id: $id) { __typename id userId scene algorithm score status clickCount conversionCount impressionCount } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("recommendationMetrics", env -> proxy(env, "recommendation",
                                "recommendationMetrics",
                                "{ recommendationMetrics(userId: $userId) { __typename totalRecommendations clickedCount purchasedCount clickThroughRate conversionRate } }",
                                Map.of("userId", env.getArgument("userId"))))
                        .dataFetcher("userBehaviorHistory", env -> proxy(env, "recommendation", "userBehaviorHistory",
                                "{ userBehaviorHistory(userId: $userId, limit: $limit) { __typename id userId productId behaviorType behaviorValue timestamp } }",
                                Map.of("userId", env.getArgument("userId"), "limit", env.getArgument("limit"))))
                        .dataFetcher("rulesByScene", env -> proxy(env, "recommendation", "rulesByScene",
                                "{ rulesByScene(scene: $scene) { __typename id ruleName scene type status priority maxItems conditions } }",
                                Map.of("scene", env.getArgument("scene")))))
                // ── Cart ───────────────────────────────────────────────────
                .type("Query", wiring -> wiring
                        .dataFetcher("cart", env -> proxy(env, "cart", "cart",
                                "{ cart(userId: $userId) { __typename id userId items { __typename id productId quantity price subtotal } totalAmount itemCount } }",
                                Map.of("userId", env.getArgument("userId")))))
                // ── Mutations ──────────────────────────────────────────────
                .type("Mutation", wiring -> wiring
                        .dataFetcher("createProduct", env -> proxy(env, "product", "createProduct",
                                "{ createProduct(input: $input) { __typename id name sku price stock status } }",
                                Map.of("input", env.getArgument("input"))))
                        .dataFetcher("updateProduct", env -> proxy(env, "product", "updateProduct",
                                "{ updateProduct(id: $id, input: $input) { __typename id name price stock status } }",
                                Map.of("id", env.getArgument("id"), "input", env.getArgument("input"))))
                        .dataFetcher("deleteProduct", env -> proxy(env, "product", "deleteProduct",
                                "mutation($id: ID!) { deleteProduct(id: $id) }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("createOrder", env -> proxy(env, "order", "createOrder",
                                "{ createOrder(input: $input) { __typename id orderNo userId totalAmount status createdAt } }",
                                Map.of("input", env.getArgument("input"))))
                        .dataFetcher("cancelOrder", env -> proxy(env, "order", "cancelOrder",
                                "{ cancelOrder(id: $id) { __typename id status } }",
                                Map.of("id", env.getArgument("id"))))
                        .dataFetcher("payOrder", env -> proxy(env, "order", "payOrder",
                                "{ payOrder(id: $id, paymentMethod: $pm) { __typename id status paymentMethod } }",
                                Map.of("id", env.getArgument("id"), "pm", env.getArgument("paymentMethod"))))
                        .dataFetcher("addToCart", env -> proxy(env, "cart", "addToCart",
                                "{ addToCart(productId: $productId, quantity: $qty) { __typename id userId items { __typename id productId quantity price subtotal } totalAmount itemCount } }",
                                Map.of("productId", env.getArgument("productId"), "qty", env.getArgument("quantity"))))
                        .dataFetcher("removeFromCart", env -> proxy(env, "cart", "removeFromCart",
                                "{ removeFromCart(cartItemId: $id) { __typename id userId items { __typename id productId quantity price subtotal } totalAmount itemCount } }",
                                Map.of("id", env.getArgument("cartItemId"))))
                        .dataFetcher("clearCart", env -> proxy(env, "cart", "clearCart",
                                "{ clearCart { __typename id userId items { __typename id productId quantity price subtotal } totalAmount itemCount } }",
                                Map.of()))
                        .dataFetcher("generateRecommendation", env -> proxy(env, "recommendation",
                                "generateRecommendation",
                                "{ generateRecommendation(userId: $userId, scene: $scene, algorithm: $algorithm, maxItems: $maxItems) { __typename id userId scene algorithm status } }",
                                Map.of("userId", env.getArgument("userId"), "scene", env.getArgument("scene"),
                                        "algorithm", env.getArgument("algorithm"), "maxItems",
                                        env.getArgument("maxItems"))))
                        .dataFetcher("recordBehavior", env -> proxy(env, "recommendation", "recordBehavior",
                                "mutation($userId: ID!, $skuCode: String!, $behaviorType: String!) { recordBehavior(userId: $userId, skuCode: $skuCode, behaviorType: $behaviorType) }",
                                Map.of("userId", env.getArgument("userId"), "skuCode", env.getArgument("skuCode"),
                                        "behaviorType", env.getArgument("behaviorType"))))
                        .dataFetcher("createRecommendationRule", env -> proxy(env, "recommendation",
                                "createRecommendationRule",
                                "{ createRecommendationRule(ruleName: $ruleName, scene: $scene, type: $type) { __typename id ruleName scene type status } }",
                                Map.of("ruleName", env.getArgument("ruleName"), "scene", env.getArgument("scene"),
                                        "type", env.getArgument("type"))))
                        .dataFetcher("activateRecommendationRule",
                                env -> proxy(env, "recommendation", "activateRecommendationRule",
                                        "mutation($id: ID!) { activateRecommendationRule(id: $id) }",
                                        Map.of("id", env.getArgument("id"))))
                        .dataFetcher("deleteRecommendationRule",
                                env -> proxy(env, "recommendation", "deleteRecommendationRule",
                                        "mutation($id: ID!) { deleteRecommendationRule(id: $id) }",
                                        Map.of("id", env.getArgument("id"))))
                        .dataFetcher("markRecommendationClicked",
                                env -> proxy(env, "recommendation", "markRecommendationClicked",
                                        "mutation($id: ID!) { markRecommendationClicked(id: $id) }",
                                        Map.of("id", env.getArgument("id"))))
                        .dataFetcher("markRecommendationPurchased",
                                env -> proxy(env, "recommendation", "markRecommendationPurchased",
                                        "mutation($id: ID!) { markRecommendationPurchased(id: $id) }",
                                        Map.of("id", env.getArgument("id")))))
                .build();
    }

    /**
     * Proxy a field query to the owning subgraph and extract the field value
     * from the JSON response.
     */
    private Object proxy(DataFetchingEnvironment env, String subgraph, String fieldName, String subgraphQuery,
            Map<String, Object> variables) {
        String url = SUBGRAPH_URLS.get(subgraph);
        if (url == null) {
            log.warn("No URL for subgraph: {}", subgraph);
            return null;
        }
        Map<String, Object> resp = executeOnSubgraph(url, subgraphQuery, variables);
        if (resp == null)
            return null;
        Object data = resp.get("data");
        if (data instanceof Map) {
            return ((Map<?, ?>) data).get(fieldName);
        }
        return null;
    }

    /**
     * Build the supergraph SDL that feeds Federation.transform().
     *
     * This is a minimal supergraph containing:
     * - _service { sdl } and _entities queries (Federation infrastructure)
     * - All root Query/Mutation fields that the gateway exposes
     * - All entity types annotated with their @key directives
     *
     * The supergraph schema (types, input types, connection types) is defined
     * in src/main/resources/graphql/schema.graphqls and is loaded by the
     * supergraph merging pipeline. Here we build only the bare minimum
     * required for Federation.transform() to inject its entity resolvers.
     */
    private String buildCombinedSdl() {
        StringBuilder sb = new StringBuilder();
        // Federation v2 service SDL type (required by Federation.transform)
        sb.append("type _Service { sdl: String! }\n");
        sb.append("type _Entity {}\n");
        sb.append("union _EntityOrUser = _Entity\n");

        // Root types — just signatures; concrete types live in schema.graphqls
        sb.append("type Query {\n");
        sb.append("  _service: _Service!\n");
        sb.append("  _entities(representations: [_Any!]!): [_Entity]!\n");
        sb.append("  product(id: ID!): Product\n");
        sb.append("  products(page: Int, size: Int): ProductConnection\n");
        sb.append("  productBySku(sku: String!): Product\n");
        sb.append("  category(id: ID!): Category\n");
        sb.append("  categories(parentId: ID): CategoryConnection\n");
        sb.append("  order(id: ID!): Order\n");
        sb.append("  orders(page: Int, size: Int, status: String): OrderConnection\n");
        sb.append("  orderByOrderNo(orderNo: String!): Order\n");
        sb.append("  user(id: ID!): User\n");
        sb.append("  users(page: Int, size: Int): UserConnection\n");
        sb.append("  inventory(productId: ID!): Inventory\n");
        sb.append("  inventoryAlerts(status: String): [InventoryAlert]\n");
        sb.append("  recommendations(userId: ID!, limit: Int): [RecommendationResult]\n");
        sb.append("  recommendationsByScene(userId: ID!, scene: String!, limit: Int): [RecommendationResult]\n");
        sb.append(
                "  recommendationsByAlgorithm(userId: ID!, algorithm: String!, limit: Int): [RecommendationResult]\n");
        sb.append("  recommendation(id: ID!): Recommendation\n");
        sb.append("  recommendationMetrics(userId: ID!): RecommendationMetrics\n");
        sb.append("  userBehaviorHistory(userId: ID!, limit: Int): [UserBehavior]\n");
        sb.append("  rulesByScene(scene: String!): [RecommendationRule]\n");
        sb.append("  cart(userId: ID!): Cart\n");
        sb.append("}\n");

        sb.append("type Mutation {\n");
        sb.append("  createProduct(input: CreateProductInput!): Product\n");
        sb.append("  updateProduct(id: ID!, input: UpdateProductInput!): Product\n");
        sb.append("  deleteProduct(id: ID!): Boolean\n");
        sb.append("  createOrder(input: CreateOrderInput!): Order\n");
        sb.append("  cancelOrder(id: ID!): Order\n");
        sb.append("  payOrder(id: ID!, paymentMethod: String!): Order\n");
        sb.append("  addToCart(productId: ID!, quantity: Int!): Cart\n");
        sb.append("  removeFromCart(cartItemId: ID!): Cart\n");
        sb.append("  clearCart: Cart\n");
        sb.append(
                "  generateRecommendation(userId: ID!, scene: String!, algorithm: String!, maxItems: Int): Recommendation\n");
        sb.append("  recordBehavior(userId: ID!, skuCode: String!, behaviorType: String!): Boolean\n");
        sb.append("  createRecommendationRule(ruleName: String!, scene: String!, type: String!): RecommendationRule\n");
        sb.append("  activateRecommendationRule(id: ID!): Boolean\n");
        sb.append("  deleteRecommendationRule(id: ID!): Boolean\n");
        sb.append("  markRecommendationClicked(id: ID!): Boolean\n");
        sb.append("  markRecommendationPurchased(id: ID!): Boolean\n");
        sb.append("}\n");

        // Entity types with @key directives
        sb.append(
                "type Product @key(fields: \"id\") { id: ID! name: String! sku: String! price: Float! stock: Int! }\n");
        sb.append("type Category @key(fields: \"id\") { id: ID! name: String! parentId: ID }\n");
        sb.append(
                "type Order @key(fields: \"id\") { id: ID! orderNo: String! userId: ID! totalAmount: Float! status: String }\n");
        sb.append(
                "type OrderItem @key(fields: \"id\") { id: ID! productId: ID! quantity: Int! price: Float! subtotal: Float! }\n");
        sb.append("type User @key(fields: \"id\") { id: ID! username: String! nickname: String email: String }\n");
        sb.append(
                "type Inventory @key(fields: \"productId\") { productId: ID! quantity: Int! availableQuantity: Int! }\n");
        sb.append(
                "type InventoryAlert @key(fields: \"id\") { id: ID! productId: ID! alertType: String! status: String! }\n");
        sb.append(
                "type Recommendation @key(fields: \"id\") { id: ID! userId: ID! scene: String! algorithm: String! status: String }\n");
        sb.append(
                "type RecommendationResult @key(fields: \"id\") { id: ID! userId: ID! productId: ID! score: Float algorithm: String }\n");
        sb.append(
                "type RecommendationRule @key(fields: \"id\") { id: ID! ruleName: String! scene: String! type: String status: String }\n");
        sb.append(
                "type UserBehavior @key(fields: \"id\") { id: ID! userId: ID! productId: ID! behaviorType: String! }\n");
        sb.append("type Cart @key(fields: \"id\") { id: ID! userId: ID! totalAmount: Float! itemCount: Int! }\n");
        sb.append(
                "type CartItem @key(fields: \"id\") { id: ID! productId: ID! quantity: Int! price: Float! subtotal: Float! }\n");

        // Connection / pagination types
        sb.append("type ProductConnection { edges: [ProductEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type ProductEdge { node: Product cursor: String }\n");
        sb.append("type OrderConnection { edges: [OrderEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type OrderEdge { node: Order cursor: String }\n");
        sb.append("type UserConnection { edges: [UserEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type UserEdge { node: User cursor: String }\n");
        sb.append("type CategoryConnection { edges: [CategoryEdge] pageInfo: PageInfo totalCount: Int }\n");
        sb.append("type CategoryEdge { node: Category cursor: String }\n");
        sb.append(
                "type PageInfo { hasNextPage: Boolean! hasPreviousPage: Boolean! startCursor: String endCursor: String }\n");
        sb.append(
                "type RecommendationMetrics { totalRecommendations: Long! clickedCount: Long! purchasedCount: Long! clickThroughRate: Float! conversionRate: Float! }\n");
        sb.append("type RecommendedItem { skuCode: String! skuName: String score: Float rank: Int reason: String }\n");

        // Input types
        sb.append(
                "input CreateProductInput { name: String! sku: String! price: Float! stock: Int! categoryId: ID description: String images: [String] }\n");
        sb.append(
                "input UpdateProductInput { name: String price: Float stock: Int categoryId: ID description: String images: [String] status: String }\n");
        sb.append(
                "input CreateOrderInput { items: [OrderItemInput!]! shippingAddress: String! paymentMethod: String! }\n");
        sb.append("input OrderItemInput { productId: ID! quantity: Int! }\n");

        return sb.toString();
    }
}
