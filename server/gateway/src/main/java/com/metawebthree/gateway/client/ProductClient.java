package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class ProductClient {

    @DubboReference
    private ProductService productService;

    /**
     * Get product by ID
     * @param id product ID
     * @return product data map
     */
    public Map<String, Object> getProductById(String id) {
        try {
            GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                    .setProductId(Long.parseLong(id))
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);
            
            Map<String, Object> result = new HashMap<>();
            ProductDetailProto product = response.getProduct();
            if (product != null) {
                result.put("id", product.getId());
                result.put("name", product.getName());
                result.put("sku", ""); // No SKU field in ProductDetailProto
                result.put("price", product.getPrice());
                result.put("stock", 0); // No stock field in ProductDetailProto
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get product by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get product by SKU
     * @param sku product SKU
     * @return product data map
     */
    public Map<String, Object> getProductBySku(String sku) {
        try {
            // GetProductDetailRequest doesn't have setSku, search by iterating
            // This is a workaround - ideally there should be a separate RPC for SKU lookup
            GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                    .setProductId(0)
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);
            
            Map<String, Object> result = new HashMap<>();
            // ProductDetailProto doesn't have SKU, return empty for now
            return result;
        } catch (Exception e) {
            log.error("Failed to get product by sku: {}, error: {}", sku, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get products with pagination - not fully implemented via Dubbo
     * @param page page number
     * @param size page size
     * @return products connection
     */
    public Map<String, Object> getProducts(int page, int size) {
        log.warn("getProducts via Dubbo not fully implemented - requires list RPC");
        return createEmptyProductsConnection(page, size);
    }

    /**
     * Create product - not implemented via Dubbo yet
     * @param input product input
     * @return created product
     */
    public Map<String, Object> createProduct(Map<String, Object> input) {
        log.warn("createProduct via Dubbo not implemented - requires REST fallback");
        return new HashMap<>();
    }

    /**
     * Update product - not implemented via Dubbo yet
     * @param id product ID
     * @param input product input
     * @return true if success
     */
    public boolean updateProduct(String id, Map<String, Object> input) {
        log.warn("updateProduct via Dubbo not implemented - requires REST fallback");
        return false;
    }

    /**
     * Delete product - not implemented via Dubbo yet
     * @param id product ID
     * @return true if success
     */
    public boolean deleteProduct(String id) {
        log.warn("deleteProduct via Dubbo not implemented - requires REST fallback");
        return false;
    }

    private Map<String, Object> createEmptyProductsConnection(int page, int size) {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", new ArrayList<>());
        connection.put("totalCount", 0);
        connection.put("pageInfo", Map.of(
            "hasNextPage", false,
            "hasPreviousPage", page > 0
        ));
        return connection;
    }
}