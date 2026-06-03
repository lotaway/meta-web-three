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
            return buildProductMap(response.getProduct());
        } catch (Exception e) {
            log.error("Failed to get product by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getProductBySku(String sku) {
        try {
            GetProductBySkuRequest request = GetProductBySkuRequest.newBuilder()
                    .setSku(sku)
                    .build();
            GetProductBySkuResponse response = productService.getProductBySku(request);
            return buildProductMap(response.getProduct());
        } catch (Exception e) {
            log.error("Failed to get product by sku: {}, error: {}", sku, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getProducts(int page, int size) {
        try {
            ListProductsRequest request = ListProductsRequest.newBuilder()
                    .setPage(page)
                    .setSize(size)
                    .build();
            ListProductsResponse response = productService.listProducts(request);

            return buildProductsConnection(
                buildProductEdges(response.getProductsList()),
                response.getTotalCount(),
                response.getPage(),
                response.getSize()
            );
        } catch (Exception e) {
            log.error("Failed to get products: page={}, size={}, error: {}", page, size, e.getMessage());
        }
        return createEmptyProductsConnection(page, size);
    }

    public Map<String, Object> createProduct(Map<String, Object> input) {
        try {
            CreateProductResponse response = productService.createProduct(buildCreateProductRequest(input));
            Map<String, Object> result = new HashMap<>();
            result.put("id", response.getId());
            result.put("success", response.getSuccess());
            result.put("message", response.getMessage());
            return result;
        } catch (Exception e) {
            log.error("Failed to create product, error: {}", e.getMessage());
        }
        return new HashMap<>();
    }

    public boolean updateProduct(String id, Map<String, Object> input) {
        try {
            UpdateProductResponse response = productService.updateProduct(buildUpdateProductRequest(id, input));
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to update product: id={}, error: {}", id, e.getMessage());
        }
        return false;
    }

    public boolean deleteProduct(String id) {
        try {
            DeleteProductRequest request = DeleteProductRequest.newBuilder()
                    .setId(Long.parseLong(id))
                    .build();
            DeleteProductResponse response = productService.deleteProduct(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to delete product: id={}, error: {}", id, e.getMessage());
        }
        return false;
    }

    private Map<String, Object> buildProductMap(ProductDetailProto product) {
        Map<String, Object> result = new HashMap<>();
        if (product != null) {
            result.put("id", product.getId());
            result.put("name", product.getName());
            result.put("sku", product.getSku());
            result.put("price", product.getPrice());
            result.put("stock", product.getStock());
            result.put("status", product.getStatus());
            result.put("category_id", product.getCategoryId());
            result.put("description", product.getDescription());
        }
        return result;
    }

    private List<Map<String, Object>> buildProductEdges(List<ProductDetailProto> products) {
        List<Map<String, Object>> edges = new ArrayList<>();
        for (ProductDetailProto product : products) {
            Map<String, Object> edge = new HashMap<>();
            edge.put("node", buildProductMap(product));
            edges.add(edge);
        }
        return edges;
    }

    private Map<String, Object> buildProductsConnection(List<Map<String, Object>> edges, int totalCount, int currentPage, int pageSize) {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", edges);
        connection.put("totalCount", totalCount);
        connection.put("pageInfo", Map.of(
            "hasNextPage", (currentPage + 1) * pageSize < totalCount,
            "hasPreviousPage", currentPage > 0
        ));
        return connection;
    }

    private CreateProductRequest buildCreateProductRequest(Map<String, Object> input) {
        CreateProductRequest.Builder builder = CreateProductRequest.newBuilder();
        if (input.containsKey("name")) builder.setName((String) input.get("name"));
        if (input.containsKey("sku")) builder.setSku((String) input.get("sku"));
        if (input.containsKey("price")) builder.setPrice(((Number) input.get("price")).doubleValue());
        if (input.containsKey("stock")) builder.setStock(((Number) input.get("stock")).intValue());
        if (input.containsKey("pic")) builder.setPic((String) input.get("pic"));
        if (input.containsKey("sub_title")) builder.setSubTitle((String) input.get("sub_title"));
        if (input.containsKey("category_id")) builder.setCategoryId(((Number) input.get("category_id")).longValue());
        if (input.containsKey("description")) builder.setDescription((String) input.get("description"));
        return builder.build();
    }

    private UpdateProductRequest buildUpdateProductRequest(String id, Map<String, Object> input) {
        UpdateProductRequest.Builder builder = UpdateProductRequest.newBuilder()
                .setId(Long.parseLong(id));
        if (input.containsKey("name")) builder.setName((String) input.get("name"));
        if (input.containsKey("sku")) builder.setSku((String) input.get("sku"));
        if (input.containsKey("price")) builder.setPrice(((Number) input.get("price")).doubleValue());
        if (input.containsKey("stock")) builder.setStock(((Number) input.get("stock")).intValue());
        if (input.containsKey("pic")) builder.setPic((String) input.get("pic"));
        if (input.containsKey("sub_title")) builder.setSubTitle((String) input.get("sub_title"));
        if (input.containsKey("category_id")) builder.setCategoryId(((Number) input.get("category_id")).longValue());
        if (input.containsKey("description")) builder.setDescription((String) input.get("description"));
        if (input.containsKey("status")) builder.setStatus(((Number) input.get("status")).intValue());
        return builder.build();
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
