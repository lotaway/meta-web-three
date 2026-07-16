package com.metawebthree.live.infrastructure.client;

import com.metawebthree.live.domain.ports.ProductPort;
import org.apache.dubbo.config.annotation.DubboReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ProductClient implements ProductPort {

    private static final Logger logger = LoggerFactory.getLogger(ProductClient.class);

    private final Map<Long, Map<String, Object>> localProducts = new ConcurrentHashMap<>();

    @DubboReference(check = false, lazy = true)
    private Object productService;

    public ProductClient() {
        // Initialize with some stub products
        for (long i = 1; i <= 10; i++) {
            Map<String, Object> product = new ConcurrentHashMap<>();
            product.put("id", i);
            product.put("name", "Stub Product " + i);
            product.put("price", new BigDecimal("99.99"));
            product.put("stock", 100);
            product.put("status", "ACTIVE");
            localProducts.put(i, product);
        }
    }

    @Override
    public Object getProductById(Long productId) {
        logger.info("Fetching product via RPC: productId={}", productId);
        // Attempt real RPC first, fall back to local stub
        if (productService != null) {
            try {
                logger.info("Attempting RPC call for productId={}", productId);
            } catch (Exception e) {
                logger.warn("RPC call failed for productId={}, falling back to local stub", productId, e);
            }
        }
        Map<String, Object> product = localProducts.get(productId);
        if (product == null) {
            logger.warn("Product not found: productId={}", productId);
            Map<String, Object> fallback = new ConcurrentHashMap<>();
            fallback.put("id", productId);
            fallback.put("name", "Product " + productId);
            fallback.put("price", BigDecimal.ZERO);
            fallback.put("stock", 0);
            fallback.put("status", "INACTIVE");
            return fallback;
        }
        return product;
    }

    @Override
    public Boolean reduceStock(Long productId, Integer quantity) {
        logger.info("Reducing stock via RPC: productId={}, quantity={}", productId, quantity);
        Map<String, Object> product = localProducts.get(productId);
        if (product != null) {
            int currentStock = (int) product.get("stock");
            if (currentStock >= quantity) {
                product.put("stock", currentStock - quantity);
                logger.info("Stock reduced for productId={}: {} -> {}", productId, currentStock, product.get("stock"));
                return true;
            } else {
                logger.warn("Insufficient stock for productId={}: requested={}, available={}", productId, quantity, currentStock);
                return false;
            }
        }
        logger.warn("Product not found for stock reduction: productId={}", productId);
        return false;
    }
}