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
                result.put("sku", "");
                result.put("price", product.getPrice());
                result.put("stock", 0);
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get product by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getProductBySku(String sku) {
        try {
            GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                    .setProductId(0)
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);

            Map<String, Object> result = new HashMap<>();
            return result;
        } catch (Exception e) {
            log.error("Failed to get product by sku: {}, error: {}", sku, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getProducts(int page, int size) {
        log.warn("getProducts via Dubbo not fully implemented - requires list RPC");
        return createEmptyProductsConnection(page, size);
    }

    public Map<String, Object> createProduct(Map<String, Object> input) {
        log.warn("createProduct via Dubbo not implemented - requires REST fallback to POST /api/products with create product proto/RPC");
        return new HashMap<>();
    }

    public boolean updateProduct(String id, Map<String, Object> input) {
        log.warn("updateProduct via Dubbo not implemented - requires REST fallback to PUT /api/products/{} with update product proto/RPC", id);
        return false;
    }

    public boolean deleteProduct(String id) {
        log.warn("deleteProduct via Dubbo not implemented - requires REST fallback to DELETE /api/products/{} with delete product proto/RPC", id);
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
