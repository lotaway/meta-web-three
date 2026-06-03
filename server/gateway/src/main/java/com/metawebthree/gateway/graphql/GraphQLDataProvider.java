package com.metawebthree.gateway.graphql;

import com.metawebthree.gateway.client.*;
import graphql.schema.DataFetchingEnvironment;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class GraphQLDataProvider {

    private final ProductClient productClient;
    private final OrderClient orderClient;
    private final UserClient userClient;
    private final CategoryClient categoryClient;
    private final InventoryClient inventoryClient;

    public GraphQLDataProvider(ProductClient productClient, OrderClient orderClient,
                                UserClient userClient, CategoryClient categoryClient,
                                InventoryClient inventoryClient) {
        this.productClient = productClient;
        this.orderClient = orderClient;
        this.userClient = userClient;
        this.categoryClient = categoryClient;
        this.inventoryClient = inventoryClient;
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
        throw new UnsupportedOperationException("Cart operations require CartService - not yet implemented");
    }

    public Object removeFromCart(DataFetchingEnvironment env) {
        throw new UnsupportedOperationException("Cart operations require CartService - not yet implemented");
    }

    public Object clearCart(DataFetchingEnvironment env) {
        throw new UnsupportedOperationException("Cart operations require CartService - not yet implemented");
    }
}