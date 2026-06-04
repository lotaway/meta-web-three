package com.metawebthree.gateway.client;

import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.common.utils.ValidationUtils;
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
                    .setProductId(ValidationUtils.parseLong(id, "id"))
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);
            return buildProductMap(response.getProduct());
        } catch (Exception e) {
            log.error("Failed to get product by id: {}, error: {}", id, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to get product by id: " + id, e);
        }
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
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to get product by sku: " + sku, e);
        }
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
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to get products", e);
        }
    }

    public Map<String, Object> createProduct(Map<String, Object> input) {
        try {
            if (input == null) throw new IllegalArgumentException("input must not be null");
            CreateProductResponse response = productService.createProduct(buildCreateProductRequest(input));
            Map<String, Object> result = new HashMap<>();
            result.put("id", response.getId());
            result.put("success", response.getSuccess());
            result.put("message", response.getMessage());
            return result;
        } catch (Exception e) {
            log.error("Failed to create product, error: {}", e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to create product", e);
        }
    }

    public boolean updateProduct(String id, Map<String, Object> input) {
        try {
            if (input == null) throw new IllegalArgumentException("input must not be null");
            UpdateProductResponse response = productService.updateProduct(buildUpdateProductRequest(id, input));
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to update product: id={}, error: {}", id, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to update product: " + id, e);
        }
    }

    public boolean deleteProduct(String id) {
        try {
            DeleteProductRequest request = DeleteProductRequest.newBuilder()
                    .setId(ValidationUtils.parseLong(id, "id"))
                    .build();
            DeleteProductResponse response = productService.deleteProduct(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to delete product: id={}, error: {}", id, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to delete product: " + id, e);
        }
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
        if (input.containsKey("name")) builder.setName(ValidationUtils.requireNonBlank((String) input.get("name"), "name"));
        if (input.containsKey("sku")) builder.setSku(ValidationUtils.requireNonBlank((String) input.get("sku"), "sku"));
        if (input.containsKey("price")) builder.setPrice(((Number) input.get("price")).doubleValue());
        if (input.containsKey("stock")) builder.setStock(ValidationUtils.parseInt(input.get("stock"), "stock"));
        Object picVal = input.get("pic");
        if (picVal instanceof String s) builder.setPic(s);
        Object subTitleVal = input.get("sub_title");
        if (subTitleVal instanceof String s) builder.setSubTitle(s);
        if (input.containsKey("category_id")) builder.setCategoryId(ValidationUtils.parseLongSafe(input.get("category_id"), "category_id"));
        Object descVal = input.get("description");
        if (descVal instanceof String s) builder.setDescription(s);
        return builder.build();
    }

    private UpdateProductRequest buildUpdateProductRequest(String id, Map<String, Object> input) {
        UpdateProductRequest.Builder builder = UpdateProductRequest.newBuilder()
                .setId(ValidationUtils.parseLong(id, "id"));
        Object nameVal = input.get("name");
        if (nameVal instanceof String s) builder.setName(s);
        Object skuVal = input.get("sku");
        if (skuVal instanceof String s) builder.setSku(s);
        if (input.containsKey("price")) builder.setPrice(((Number) input.get("price")).doubleValue());
        if (input.containsKey("stock")) builder.setStock(ValidationUtils.parseInt(input.get("stock"), "stock"));
        Object picVal = input.get("pic");
        if (picVal instanceof String s) builder.setPic(s);
        Object subTitleVal = input.get("sub_title");
        if (subTitleVal instanceof String s) builder.setSubTitle(s);
        if (input.containsKey("category_id")) builder.setCategoryId(ValidationUtils.parseLongSafe(input.get("category_id"), "category_id"));
        Object descVal = input.get("description");
        if (descVal instanceof String s) builder.setDescription(s);
        if (input.containsKey("status")) builder.setStatus(ValidationUtils.parseInt(input.get("status"), "status"));
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
