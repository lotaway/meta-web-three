package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.entity.ApiSubscription;
import com.metawebthree.developerportal.entity.ApiSubscription.SubscriptionStatus;
import com.metawebthree.developerportal.repository.ApiSubscriptionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiDocumentationService {

    public Map<String, Object> generateOpenApiDocumentation(String baseUrl) {
        Map<String, Object> openapi = new LinkedHashMap<>();
        openapi.put("openapi", "3.0.0");

        Map<String, Object> info = new LinkedHashMap<>();
        info.put("title", "Meta Web Three API Platform");
        info.put("description", "Comprehensive API platform for e-commerce, AI, blockchain, and enterprise services");
        info.put("version", "1.0.0");
        info.put("contact", Map.of(
            "name", "API Support",
            "email", "api@metawebthree.com",
            "url", "https://developer.metawebthree.com"
        ));
        openapi.put("info", info);

        List<Map<String, String>> servers = new ArrayList<>();
        servers.add(Map.of(
            "url", baseUrl != null ? baseUrl : "https://api.metawebthree.com",
            "description", "Production server"
        ));
        servers.add(Map.of(
            "url", "https://sandbox-api.metawebthree.com",
            "description", "Sandbox server for testing"
        ));
        openapi.put("servers", servers);

        Map<String, Object> paths = generatePaths();
        openapi.put("paths", paths);

        Map<String, Object> components = generateComponents();
        openapi.put("components", components);

        List<Map<String, List<String>>> security = new ArrayList<>();
        security.add(Map.of("ApiKeyAuth", Collections.emptyList()));
        security.add(Map.of("OAuth2", Arrays.asList("read", "write")));
        openapi.put("security", security);

        List<Map<String, String>> tags = generateTags();
        openapi.put("tags", tags);

        return openapi;
    }

    private Map<String, Object> generatePaths() {
        Map<String, Object> paths = new LinkedHashMap<>();

        addPath(paths, "/user-service/api/v1/users/{userId}", "get", Map.of(
            "summary", "Get user by ID",
            "tags", Collections.singletonList("User Service"),
            "description", "Retrieve user information by user ID",
            "parameters", Arrays.asList(
                Map.of("name", "userId", "in", "path", "required", true, "schema", Map.of("type", "string"))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Success", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/User")))),
                "404", Map.of("description", "User not found")
            )
        ));

        addPath(paths, "/product-service/api/v1/products", "get", Map.of(
            "summary", "List products",
            "tags", Collections.singletonList("Product Service"),
            "description", "Retrieve a list of products with optional filters",
            "parameters", Arrays.asList(
                Map.of("name", "page", "in", "query", "schema", Map.of("type", "integer")),
                Map.of("name", "size", "in", "query", "schema", Map.of("type", "integer")),
                Map.of("name", "categoryId", "in", "query", "schema", Map.of("type", "string"))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Success", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/ProductList"))))
            )
        ));

        addPath(paths, "/product-service/api/v1/products/{productId}", "get", Map.of(
            "summary", "Get product details",
            "tags", Collections.singletonList("Product Service"),
            "description", "Retrieve detailed product information",
            "parameters", Arrays.asList(
                Map.of("name", "productId", "in", "path", "required", true, "schema", Map.of("type", "string"))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Success", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/Product")))),
                "404", Map.of("description", "Product not found")
            )
        ));

        addPath(paths, "/order-service/api/v1/orders", "get", Map.of(
            "summary", "List orders",
            "tags", Collections.singletonList("Order Service"),
            "description", "Retrieve orders for authenticated user",
            "parameters", Arrays.asList(
                Map.of("name", "page", "in", "query", "schema", Map.of("type", "integer")),
                Map.of("name", "size", "in", "query", "schema", Map.of("type", "integer")),
                Map.of("name", "status", "in", "query", "schema", Map.of("type", "string"))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Success", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/OrderList"))))
            )
        ));

        addPath(paths, "/order-service/api/v1/orders", "post", Map.of(
            "summary", "Create order",
            "tags", Collections.singletonList("Order Service"),
            "description", "Create a new order",
            "requestBody", Map.of(
                "required", true,
                "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/CreateOrderRequest")))
            ),
            "responses", Map.of(
                "201", Map.of("description", "Order created", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/Order")))),
                "400", Map.of("description", "Invalid request")
            )
        ));

        addPath(paths, "/inventory-service/api/v1/inventory/{productId}", "get", Map.of(
            "summary", "Get inventory",
            "tags", Collections.singletonList("Inventory Service"),
            "description", "Check inventory status for a product",
            "parameters", Arrays.asList(
                Map.of("name", "productId", "in", "path", "required", true, "schema", Map.of("type", "string"))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Success", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/Inventory"))))
            )
        ));

        addPath(paths, "/payment-service/api/v1/payments", "post", Map.of(
            "summary", "Process payment",
            "tags", Collections.singletonList("Payment Service"),
            "description", "Process a payment for an order",
            "requestBody", Map.of(
                "required", true,
                "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/PaymentRequest")))
            ),
            "responses", Map.of(
                "200", Map.of("description", "Payment processed", "content", Map.of("application/json", Map.of("schema", Map.of("$ref", "#/components/schemas/PaymentResult")))),
                "400", Map.of("description", "Payment failed")
            )
        ));

        return paths;
    }

    private void addPath(Map<String, Object> paths, String path, String method, Map<String, Object> operation) {
        if (!paths.containsKey(path)) {
            paths.put(path, new LinkedHashMap<>());
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> pathItem = (Map<String, Object>) paths.get(path);
        pathItem.put(method, operation);
    }

    private Map<String, Object> generateComponents() {
        Map<String, Object> components = new LinkedHashMap<>();

        Map<String, Object> securitySchemes = new LinkedHashMap<>();

        securitySchemes.put("ApiKeyAuth", Map.of(
            "type", "apiKey",
            "in", "header",
            "name", "X-API-Key",
            "description", "API Key authentication"
        ));

        securitySchemes.put("OAuth2", Map.of(
            "type", "oauth2",
            "flows", Map.of(
                "authorizationCode", Map.of(
                    "authorizationUrl", "https://api.metawebthree.com/oauth/authorize",
                    "tokenUrl", "https://api.metawebthree.com/oauth/token",
                    "scopes", Map.of(
                        "read", "Read access",
                        "write", "Write access",
                        "admin", "Admin access"
                    )
                ),
                "clientCredentials", Map.of(
                    "tokenUrl", "https://api.metawebthree.com/oauth/token",
                    "scopes", Map.of(
                        "read", "Read access",
                        "write", "Write access"
                    )
                )
            )
        ));

        components.put("securitySchemes", securitySchemes);

        Map<String, Object> schemas = new LinkedHashMap<>();

        schemas.put("User", Map.of(
            "type", "object",
            "properties", Map.of(
                "userId", Map.of("type", "string"),
                "username", Map.of("type", "string"),
                "email", Map.of("type", "string"),
                "phone", Map.of("type", "string"),
                "createdAt", Map.of("type", "string", "format", "date-time")
            )
        ));

        schemas.put("Product", Map.of(
            "type", "object",
            "properties", Map.of(
                "productId", Map.of("type", "string"),
                "name", Map.of("type", "string"),
                "description", Map.of("type", "string"),
                "price", Map.of("type", "number"),
                "categoryId", Map.of("type", "string"),
                "stock", Map.of("type", "integer")
            )
        ));

        schemas.put("ProductList", Map.of(
            "type", "object",
            "properties", Map.of(
                "content", Map.of("type", "array", "items", Map.of("$ref", "#/components/schemas/Product")),
                "totalElements", Map.of("type", "integer"),
                "totalPages", Map.of("type", "integer"),
                "size", Map.of("type", "integer"),
                "number", Map.of("type", "integer")
            )
        ));

        schemas.put("Order", Map.of(
            "type", "object",
            "properties", Map.of(
                "orderId", Map.of("type", "string"),
                "userId", Map.of("type", "string"),
                "status", Map.of("type", "string"),
                "totalAmount", Map.of("type", "number"),
                "items", Map.of("type", "array", "items", Map.of("$ref", "#/components/schemas/OrderItem")),
                "createdAt", Map.of("type", "string", "format", "date-time")
            )
        ));

        schemas.put("OrderItem", Map.of(
            "type", "object",
            "properties", Map.of(
                "productId", Map.of("type", "string"),
                "productName", Map.of("type", "string"),
                "quantity", Map.of("type", "integer"),
                "price", Map.of("type", "number")
            )
        ));

        schemas.put("OrderList", Map.of(
            "type", "object",
            "properties", Map.of(
                "content", Map.of("type", "array", "items", Map.of("$ref", "#/components/schemas/Order")),
                "totalElements", Map.of("type", "integer"),
                "totalPages", Map.of("type", "integer")
            )
        ));

        schemas.put("CreateOrderRequest", Map.of(
            "type", "object",
            "required", Arrays.asList("items", "addressId"),
            "properties", Map.of(
                "items", Map.of("type", "array", "items", Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "productId", Map.of("type", "string"),
                        "quantity", Map.of("type", "integer")
                    )
                )),
                "addressId", Map.of("type", "string"),
                "couponCode", Map.of("type", "string"),
                "remark", Map.of("type", "string")
            )
        ));

        schemas.put("Inventory", Map.of(
            "type", "object",
            "properties", Map.of(
                "productId", Map.of("type", "string"),
                "totalQty", Map.of("type", "integer"),
                "availableQty", Map.of("type", "integer"),
                "reservedQty", Map.of("type", "integer")
            )
        ));

        schemas.put("PaymentRequest", Map.of(
            "type", "object",
            "required", Arrays.asList("orderId", "paymentMethod"),
            "properties", Map.of(
                "orderId", Map.of("type", "string"),
                "paymentMethod", Map.of("type", "string", "enum", Arrays.asList("ALIPAY", "WECHAT", "BANK_CARD")),
                "amount", Map.of("type", "number")
            )
        ));

        schemas.put("PaymentResult", Map.of(
            "type", "object",
            "properties", Map.of(
                "paymentId", Map.of("type", "string"),
                "orderId", Map.of("type", "string"),
                "status", Map.of("type", "string"),
                "transactionId", Map.of("type", "string"),
                "paidAt", Map.of("type", "string", "format", "date-time")
            )
        ));

        schemas.put("Error", Map.of(
            "type", "object",
            "properties", Map.of(
                "code", Map.of("type", "string"),
                "message", Map.of("type", "string"),
                "timestamp", Map.of("type", "string", "format", "date-time"),
                "path", Map.of("type", "string")
            )
        ));

        components.put("schemas", schemas);

        return components;
    }

    private List<Map<String, String>> generateTags() {
        List<Map<String, String>> tags = new ArrayList<>();

        tags.add(Map.of(
            "name", "User Service",
            "description", "User management and authentication APIs"
        ));

        tags.add(Map.of(
            "name", "Product Service",
            "description", "Product catalog and category management APIs"
        ));

        tags.add(Map.of(
            "name", "Order Service",
            "description", "Order creation and management APIs"
        ));

        tags.add(Map.of(
            "name", "Inventory Service",
            "description", "Inventory management and stock check APIs"
        ));

        tags.add(Map.of(
            "name", "Payment Service",
            "description", "Payment processing APIs"
        ));

        return tags;
    }

    private final ApiSubscriptionRepository apiSubscriptionRepository;

    public Map<String, Object> generatePersonalizedDocumentation(String developerId, String baseUrl) {
        Map<String, Object> fullDoc = generateOpenApiDocumentation(baseUrl);

        List<ApiSubscription> activeSubscriptions = apiSubscriptionRepository
                .findByDeveloperIdAndStatus(developerId, SubscriptionStatus.ACTIVE);

        Set<String> allowedPatterns = activeSubscriptions.stream()
                .map(ApiSubscription::getApiPattern)
                .collect(Collectors.toSet());

        @SuppressWarnings("unchecked")
        Map<String, Object> paths = (Map<String, Object>) fullDoc.get("paths");
        Map<String, Object> filteredPaths = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : paths.entrySet()) {
            String path = entry.getKey();
            boolean matched = allowedPatterns.isEmpty() || allowedPatterns.stream()
                    .anyMatch(pattern -> pathMatchesPattern(path, pattern));
            if (matched) {
                filteredPaths.put(entry.getKey(), entry.getValue());
            }
        }
        fullDoc.put("paths", filteredPaths);

        return fullDoc;
    }

    private boolean pathMatchesPattern(String path, String pattern) {
        if (pattern.endsWith("/**")) {
            String prefix = pattern.substring(0, pattern.length() - 3);
            return path.startsWith(prefix);
        } else if (pattern.endsWith("/*")) {
            String prefix = pattern.substring(0, pattern.length() - 2);
            return path.startsWith(prefix) && !path.substring(prefix.length()).contains("/");
        }
        return path.equals(pattern);
    }

    public Map<String, String> generateSdkSamples(String language) {
        Map<String, String> samples = new LinkedHashMap<>();

        switch (language.toLowerCase()) {
            case "java":
                samples.put("java", generateJavaSample());
                break;
            case "python":
                samples.put("python", generatePythonSample());
                break;
            case "javascript":
            case "js":
                samples.put("javascript", generateJavaScriptSample());
                break;
            case "curl":
                samples.put("curl", generateCurlSample());
                break;
            default:
                samples.put("java", generateJavaSample());
                samples.put("python", generatePythonSample());
                samples.put("javascript", generateJavaScriptSample());
                samples.put("curl", generateCurlSample());
        }

        return samples;
    }

    private String generateJavaSample() {
        return """
            // Java SDK Sample
            import com.metawebthree.api.client.MetaWebThreeClient;
            
            public class Example {
                public static void main(String[] args) {
                    MetaWebThreeClient client = MetaWebThreeClient.builder()
                        .apiKey("YOUR_API_KEY")
                        .baseUrl("https://api.metawebthree.com")
                        .build();
                    
                    // Get product
                    Product product = client.products().get("product-123");
                    System.out.println(product.getName());
                    
                    // Create order
                    Order order = client.orders().create(OrderRequest.builder()
                        .addItem("product-123", 2)
                        .addressId("address-456")
                        .build());
                }
            }
            """;
    }

    private String generatePythonSample() {
        return """
            # Python SDK Sample
            from metawebthree import MetaWebThreeClient
            
            client = MetaWebThreeClient(
                api_key="YOUR_API_KEY",
                base_url="https://api.metawebthree.com"
            )
            
            # Get product
            product = client.products.get("product-123")
            print(product.name)
            
            # Create order
            order = client.orders.create(
                items=[{"product_id": "product-123", "quantity": 2}],
                address_id="address-456"
            )
            """;
    }

    private String generateJavaScriptSample() {
        return """
            // JavaScript SDK Sample
            const { MetaWebThreeClient } = require('@metawebthree/sdk');
            
            const client = new MetaWebThreeClient({
                apiKey: 'YOUR_API_KEY',
                baseUrl: 'https://api.metawebthree.com'
            });
            
            // Get product
            const product = await client.products.get('product-123');
            console.log(product.name);
            
            // Create order
            const order = await client.orders.create({
                items: [{ productId: 'product-123', quantity: 2 }],
                addressId: 'address-456'
            });
            """;
    }

    private String generateCurlSample() {
        return """
            # cURL Sample
            
            # Get product
            curl -X GET "https://api.metawebthree.com/product-service/api/v1/products/product-123" \\
                -H "X-API-Key: YOUR_API_KEY" \\
                -H "Content-Type: application/json"
            
            # Create order
            curl -X POST "https://api.metawebthree.com/order-service/api/v1/orders" \\
                -H "X-API-Key: YOUR_API_KEY" \\
                -H "Content-Type: application/json" \\
                -d '{
                    "items": [{"productId": "product-123", "quantity": 2}],
                    "addressId": "address-456"
                }'
            """;
    }

    public Map<String, Object> generateSandboxTestData(String developerId) {
        Map<String, Object> sandboxData = new LinkedHashMap<>();

        sandboxData.put("testUsers", Arrays.asList(
            Map.of("userId", "sandbox-user-001", "username", "test_user_1", "email", "test1@sandbox.metawebthree.com"),
            Map.of("userId", "sandbox-user-002", "username", "test_user_2", "email", "test2@sandbox.metawebthree.com")
        ));

        sandboxData.put("testProducts", Arrays.asList(
            Map.of("productId", "sandbox-product-001", "name", "Sandbox Test Product 1", "price", 99.99, "stock", 100),
            Map.of("productId", "sandbox-product-002", "name", "Sandbox Test Product 2", "price", 199.99, "stock", 50),
            Map.of("productId", "sandbox-product-003", "name", "Sandbox Test Product 3", "price", 299.99, "stock", 25)
        ));

        sandboxData.put("testAddresses", Arrays.asList(
            Map.of("addressId", "sandbox-address-001", "recipient", "Test Recipient", "phone", "13800138000", 
                "province", "Guangdong", "city", "Shenzhen", "district", "Nanshan", "detail", "Test Address 123"),
            Map.of("addressId", "sandbox-address-002", "recipient", "Test Recipient 2", "phone", "13900139000",
                "province", "Beijing", "city", "Beijing", "district", "Chaoyang", "detail", "Test Road 456")
        ));

        sandboxData.put("testOrders", Arrays.asList(
            Map.of("orderId", "sandbox-order-001", "status", "PENDING", "totalAmount", 199.98,
                "items", Arrays.asList(Map.of("productId", "sandbox-product-001", "quantity", 2, "price", 99.99)))
        ));

        sandboxData.put("testPaymentMethods", Arrays.asList(
            Map.of("method", "ALIPAY", "description", "Alipay Sandbox - Test payment will always succeed"),
            Map.of("method", "WECHAT", "description", "WeChat Pay Sandbox - Test payment will always succeed"),
            Map.of("method", "BANK_CARD", "description", "Bank Card Sandbox - Use test card: 4242424242424242")
        ));

        sandboxData.put("sandboxConfig", Map.of(
            "baseUrl", "https://sandbox-api.metawebthree.com",
            "rateLimit", 1000,
            "dailyQuota", 10000,
            "features", Arrays.asList(
                "instant_order_creation",
                "mock_payment_success",
                "simulated_logistics_tracking",
                "test_refund_flow"
            )
        ));

        log.info("Generated sandbox test data for developer: {}", developerId);
        return sandboxData;
    }

    public Map<String, Object> resetSandboxEnvironment(String developerId) {
        log.info("Sandbox environment reset for developer: {}", developerId);

        return Map.of(
            "developerId", developerId,
            "status", "reset_complete",
            "message", "Sandbox environment has been reset. All test data has been cleared.",
            "newTestData", generateSandboxTestData(developerId),
            "resetAt", java.time.LocalDateTime.now().toString()
        );
    }
}
