package com.metawebthree.product.interfaces.web;

import com.metawebthree.product.application.ProductService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.constraints.NotNull;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.dto.ProductDTO;
import com.metawebthree.common.dto.ProductDetailDTO;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/products")
@Tag(name = "Product Management", description = "Endpoints for product lifecycle and management")
public class ProductController {

    private final ProductService productService;

    public ProductController(ProductService productService) {
        this.productService = productService;
    }

    @Operation(summary = "Check service status", description = "Basic health check for product service")
    @GetMapping("/health")
    public ApiResponse<String> healthCheck() {
        return ApiResponse.success("product-service is running");
    }

    @Operation(summary = "Create a new product")
    @PostMapping
    public ApiResponse<Void> createProduct() {
        productService.createProduct();
        return ApiResponse.success();
    }

    @Operation(summary = "List products with filters")
    @GetMapping
    public ApiResponse<List<ProductDTO>> listProducts(
            @Parameter(description = "Category ID") @RequestParam(required = false) Integer categoryId,
            @Parameter(description = "Search keyword") @RequestParam(required = false) String keyword,
            @Parameter(description = "Price range") @RequestParam(required = false) String priceRange) {
        return ApiResponse.success(productService.listProducts(categoryId, keyword, priceRange));
    }

    @Operation(summary = "Get detailed product info by ID")
    @GetMapping("/{id}")
    public ApiResponse<ProductDetailDTO> getProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        ProductDetailDTO detail = productService.getProductDetail(id);
        if (detail == null) {
            return ApiResponse.error("Product not found");
        }
        return ApiResponse.success(detail);
    }

    @Operation(summary = "Update product content", description = "Updates the description/content of an existing product")
    @PutMapping("/{id}")
    public ApiResponse<Void> updateProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id,
            @RequestBody String content) {
        productService.updateProduct(Long.valueOf(id), content.getBytes());
        return ApiResponse.success();
    }

    @Operation(summary = "Delete product", description = "Deletes a product by its ID")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> deleteProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        productService.deleteProduct(id.toString());
        return ApiResponse.success();
    }

    @Operation(summary = "Upload product image", description = "Uploads an image for a specific product")
    @PostMapping(path = "/{id}/images", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<Void> uploadImage(
            @Parameter(description = "Product ID") @PathVariable Integer id,
            @Parameter(description = "Image file") @RequestParam MultipartFile file) {
        productService.uploadImage(Long.valueOf(id), file);
        return ApiResponse.success();
    }
}
