package com.metawebthree.gateway.graphql;

import com.metawebthree.gateway.client.*;
import graphql.schema.DataFetchingEnvironment;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class GraphQLDataProvider {

    private final ProductClient productClient;
    private final OrderClient orderClient;
    private final UserClient userClient;
    private final CategoryClient categoryClient;
    private final InventoryClient inventoryClient;
    private final RecommendationClient recommendationClient;

    private final Map<String, Map<String, Object>> carts = new ConcurrentHashMap<>();
    private final AtomicLong cartItemIdCounter = new AtomicLong(1);

    public GraphQLDataProvider(ProductClient productClient, OrderClient orderClient,
                                UserClient userClient, CategoryClient categoryClient,
                                InventoryClient inventoryClient,
                                RecommendationClient recommendationClient) {
        this.productClient = productClient;
        this.orderClient = orderClient;
        this.userClient = userClient;
        this.categoryClient = categoryClient;
        this.inventoryClient = inventoryClient;
        this.recommendationClient = recommendationClient;
    }

    public Object getProduct(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return productClient.getProductById(id);
    }

    public Object getProducts(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        
        page = page != null ? page : 0;
        size = size != null ? size : 10;
        
        return productClient.getProducts(page, size);
    }

    public Object getProductBySku(DataFetchingEnvironment env) {
        String sku = env.getArgument("sku");
        return productClient.getProductBySku(sku);
    }

    public Object getOrder(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return orderClient.getOrderById(id);
    }

    public Object getOrders(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        
        page = page != null ? page : 0;
        size = size != null ? size : 10;
        
        return orderClient.getOrders(page, size);
    }

    public Object getOrderByOrderNo(DataFetchingEnvironment env) {
        String orderNo = env.getArgument("orderNo");
        return orderClient.getOrderByOrderNo(orderNo);
    }

    public Object getUser(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return userClient.getUserById(id);
    }

    public Object getUsers(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        
        page = page != null ? page : 0;
        size = size != null ? size : 10;
        
        return userClient.getUsers(page, size);
    }

    public Object getCategory(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return categoryClient.getCategoryById(id);
    }

    public Object getCategories(DataFetchingEnvironment env) {
        return categoryClient.getCategories();
    }

    public Object getInventory(DataFetchingEnvironment env) {
        String productId = env.getArgument("productId");
        return inventoryClient.getInventoryByProductId(productId);
    }

    public Object getInventoryAlerts(DataFetchingEnvironment env) {
        return inventoryClient.getInventoryAlerts();
    }

    public Object createOrder(DataFetchingEnvironment env) {
        Map<String, Object> input = env.getArgument("input");
        return orderClient.createOrder(input);
    }

    public Object cancelOrder(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        boolean success = orderClient.cancelOrder(id);
        Map<String, Object> order = new HashMap<>();
        order.put("id", id);
        order.put("status", success ? "CANCELLED" : "FAILED");
        order.put("updatedAt", new Date().toString());
        return order;
    }

    public Object payOrder(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        String paymentMethod = env.getArgument("paymentMethod");
        boolean success = orderClient.payOrder(id, paymentMethod);
        Map<String, Object> order = new HashMap<>();
        order.put("id", id);
        order.put("status", success ? "PAID" : "FAILED");
        order.put("paymentMethod", paymentMethod);
        order.put("updatedAt", new Date().toString());
        return order;
    }

    public Object createProduct(DataFetchingEnvironment env) {
        Map<String, Object> input = env.getArgument("input");
        return productClient.createProduct(input);
    }

    public Object updateProduct(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        Map<String, Object> input = env.getArgument("input");
        productClient.updateProduct(id, input);
        Map<String, Object> product = new HashMap<>();
        product.put("id", id);
        product.put("name", input.get("name"));
        product.put("price", input.get("price"));
        product.put("stock", input.get("stock"));
        product.put("status", input.get("status"));
        product.put("updatedAt", new Date().toString());
        return product;
    }

    public Boolean deleteProduct(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return productClient.deleteProduct(id);
    }

    public Object addToCart(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        String productId = env.getArgument("productId");
        Integer quantity = env.getArgument("quantity");
        if (quantity == null) quantity = 1;

        String cartKey = "cart:" + (userId != null ? userId : "anonymous");
        Map<String, Object> cart = carts.computeIfAbsent(cartKey, k -> {
            Map<String, Object> c = new HashMap<>();
            c.put("id", cartKey);
            c.put("userId", userId);
            c.put("items", new ArrayList<Map<String, Object>>());
            c.put("totalAmount", 0);
            c.put("itemCount", 0);
            return c;
        });

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> items = (List<Map<String, Object>>) cart.get("items");

        synchronized (items) {
            boolean found = false;
            for (Map<String, Object> item : items) {
                if (productId.equals(item.get("productId"))) {
                    int oldQty = (int) item.get("quantity");
                    item.put("quantity", oldQty + quantity);
                    found = true;
                    break;
                }
            }
            if (!found) {
                Map<String, Object> newItem = new HashMap<>();
                newItem.put("id", String.valueOf(cartItemIdCounter.incrementAndGet()));
                newItem.put("productId", productId);
                newItem.put("quantity", quantity);
                newItem.put("price", 0);
                newItem.put("subtotal", 0);
                items.add(newItem);
            }

            int totalAmount = 0;
            int itemCount = 0;
            for (Map<String, Object> item : items) {
                itemCount += (int) item.get("quantity");
                totalAmount += (int) item.get("quantity") * (int) item.get("price");
            }
            cart.put("totalAmount", totalAmount);
            cart.put("itemCount", itemCount);
        }

        return cart;
    }

    public Object removeFromCart(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        String cartItemId = env.getArgument("cartItemId");
        String cartKey = "cart:" + (userId != null ? userId : "anonymous");

        Map<String, Object> cart = carts.get(cartKey);
        if (cart == null) return Collections.singletonMap("error", "Cart not found");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> items = (List<Map<String, Object>>) cart.get("items");

        synchronized (items) {
            items.removeIf(item -> cartItemId.equals(item.get("id")));

            int totalAmount = 0;
            int itemCount = 0;
            for (Map<String, Object> item : items) {
                itemCount += (int) item.get("quantity");
                totalAmount += (int) item.get("quantity") * (int) item.get("price");
            }
            cart.put("totalAmount", totalAmount);
            cart.put("itemCount", itemCount);
        }

        return cart;
    }

    public Object clearCart(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        String cartKey = "cart:" + (userId != null ? userId : "anonymous");

        Map<String, Object> cart = carts.get(cartKey);
        if (cart == null) return Collections.singletonMap("error", "Cart not found");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> items = (List<Map<String, Object>>) cart.get("items");
        items.clear();
        cart.put("totalAmount", 0);
        cart.put("itemCount", 0);

        return cart;
    }

    // ==================== Recommendation ====================

    public Object getRecommendations(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        Integer limit = env.getArgument("limit");
        limit = limit != null ? limit : 10;
        return recommendationClient.getUserRecommendations(userId);
    }

    public Object getRecommendationsByScene(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        String scene = env.getArgument("scene");
        return recommendationClient.getUserRecommendationsByScene(userId, scene);
    }

    public Object getRecommendationsByAlgorithm(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        String algorithm = env.getArgument("algorithm");
        Integer limit = env.getArgument("limit");
        limit = limit != null ? limit : 10;
        return recommendationClient.getRecommendationsByAlgorithm(userId, algorithm, limit);
    }

    public Object getRecommendation(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return recommendationClient.getRecommendationById(parseLongId(id));
    }

    public Object getRecommendationMetrics(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        return recommendationClient.getMetrics(userId);
    }

    public Object getUserBehaviorHistory(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        Integer limit = env.getArgument("limit");
        limit = limit != null ? limit : 50;
        return recommendationClient.getBehaviorHistory(userId, limit);
    }

    public Object getRulesByScene(DataFetchingEnvironment env) {
        String scene = env.getArgument("scene");
        return recommendationClient.getRulesByScene(scene);
    }

    public Object generateRecommendation(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        String scene = env.getArgument("scene");
        String algorithm = env.getArgument("algorithm");
        Integer maxItems = env.getArgument("maxItems");
        maxItems = maxItems != null ? maxItems : 10;
        return recommendationClient.generateRecommendation(userId, scene, algorithm, maxItems);
    }

    public Boolean recordBehavior(DataFetchingEnvironment env) {
        Long userId = getUserIdRequired(env);
        String skuCode = env.getArgument("skuCode");
        String behaviorType = env.getArgument("behaviorType");
        recommendationClient.recordBehavior(userId, skuCode, behaviorType);
        return true;
    }

    public Object createRecommendationRule(DataFetchingEnvironment env) {
        String ruleName = env.getArgument("ruleName");
        String scene = env.getArgument("scene");
        String type = env.getArgument("type");
        return recommendationClient.createRule(ruleName, scene, type);
    }

    public Boolean activateRecommendationRule(DataFetchingEnvironment env) {
        Long id = parseLongId(env.getArgument("id"));
        recommendationClient.activateRule(id);
        return true;
    }

    public Boolean deleteRecommendationRule(DataFetchingEnvironment env) {
        Long id = parseLongId(env.getArgument("id"));
        recommendationClient.deleteRule(id);
        return true;
    }

    public Boolean markRecommendationClicked(DataFetchingEnvironment env) {
        Long id = parseLongId(env.getArgument("id"));
        recommendationClient.markAsClicked(id);
        return true;
    }

    public Boolean markRecommendationPurchased(DataFetchingEnvironment env) {
        Long id = parseLongId(env.getArgument("id"));
        recommendationClient.markAsPurchased(id);
        return true;
    }

    private Long getUserIdRequired(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        if (userId == null || userId.isEmpty()) {
            throw new IllegalArgumentException("userId is required");
        }
        return Long.parseLong(userId);
    }

    private Long parseLongId(String id) {
        if (id == null || id.isEmpty()) {
            throw new IllegalArgumentException("id is required");
        }
        return Long.parseLong(id);
    }
}