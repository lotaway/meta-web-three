package com.metawebthree.product;

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
@RequestMapping("/product")
@Tag(name = "Product Management")
public class ProductController {

    private final ProductService productService;

    public ProductController(ProductService productService) {
        this.productService = productService;
    }

    @Operation(summary = "Test microservice connection", description = "Simple endpoint to test if the product service is running")
    @GetMapping("/micro-service-test")
    public String microServiceTest() {
        return "product-service";
    }

    @Operation(summary = "Create a new product")
    @PostMapping("/create")
    public ApiResponse<Boolean> create() {
        return ApiResponse.success(productService.createProduct());
    }

    @Operation(summary = "Get product by ID")
    @GetMapping("/{id}")
    public ApiResponse<?> get(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        ProductDTO dto = productService.getProductById(id);
        if (dto == null) {
            return ApiResponse.error("Product not found");
        }
        return ApiResponse.success(dto);
    }

    @Operation(summary = "Update product content", description = "Updates the content of an existing product")
    @PutMapping("/{id}")
    public ApiResponse<Boolean> update(
            @Parameter(description = "Product ID") @PathVariable Long id,
            @Parameter(description = "Product content") @RequestParam String content) {
        boolean success = productService.updateProduct(id, content.getBytes());
        return ApiResponse.success(success);
    }

    @Operation(summary = "Delete product", description = "Deletes a product by its ID")
    @DeleteMapping("/{id}")
    public ApiResponse<Boolean> delete(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        productService.deleteProduct(id.toString());
        return ApiResponse.success(true);
    }

    @Operation(summary = "Get detailed product info")
    @GetMapping("/detail/{id}")
    public ApiResponse<?> getDetail(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        ProductDetailDTO detail = productService.getProductDetail(id);
        if (detail == null) {
            return ApiResponse.error("Product not found");
        }
        return ApiResponse.success(detail);
    }

    @Operation(summary = "List products with filters")
    @GetMapping("/list")
    public ApiResponse<List<ProductDTO>> list(
            @Parameter(description = "Category ID") @RequestParam(required = false) Integer categoryId,
            @Parameter(description = "Search keyword") @RequestParam(required = false) String keyword,
            @Parameter(description = "Price range") @RequestParam(required = false) String priceRange) {
        return ApiResponse.success(productService.listProducts(categoryId, keyword, priceRange));
    }

    @Operation(summary = "Upload product image", description = "Uploads an image for a specific product")
    @PostMapping(path = "/product/{id}/images", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<Boolean> uploadImage(
            @Parameter(description = "Product ID") @PathVariable(name = "id") Long productId,
            @Parameter(description = "Image file") @RequestParam MultipartFile file) {
        boolean success = productService.uploadImage(productId, file);
        return ApiResponse.success(success);
    }
}
