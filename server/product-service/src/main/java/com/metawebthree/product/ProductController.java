package com.metawebthree.product;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.apache.rocketmq.client.exception.MQBrokerException;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.remoting.exception.RemotingException;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.dto.ProductDTO;
import com.metawebthree.common.dto.ProductDetailDTO;
import java.util.List;

import java.util.Arrays;

@Slf4j
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
        return "from micro-service product-service";
    }

    @Operation(summary = "Create a new product")
    @PostMapping("/create")
    public ApiResponse<Boolean> create() {
        return ApiResponse.success(productService.createProduct());
    }

    @Operation(summary = "Get product by ID")
    @GetMapping("/{id}")
    public ApiResponse<ProductDTO> get(@PathVariable Integer id) {
        ProductDTO dto = new ProductDTO();
        dto.setId(id);
        dto.setName("name");
        return ApiResponse.success(dto);
    }

    @Operation(summary = "Update product content", description = "Updates the content of an existing product")
    @PutMapping("/{id}")
    public String update(@PathVariable Long id, @RequestParam String content) {
        Boolean res = productService.updateProduct(id, content.getBytes());
        return res.toString();
    }

    @Operation(summary = "Delete product", description = "Deletes a product by its ID")
    @DeleteMapping("/{id}")
    public boolean delete(@PathVariable Integer id)
            throws MQBrokerException, RemotingException, InterruptedException, MQClientException {
        productService.deleteProduct("test.txt");
        return true;
    }

    @Operation(summary = "Get detailed product info")
    @GetMapping("/detail/{id}")
    public ApiResponse<ProductDetailDTO> getDetail(@PathVariable Integer id) {
        ProductDetailDTO detail = productService.getProductDetail(id);
        if (detail == null) {
            return ApiResponse.error("Product not found");
        }
        return ApiResponse.success(detail);
    }

    @Operation(summary = "List products with filters")
    @GetMapping("/list")
    public ApiResponse<List<ProductDTO>> list(
            @RequestParam(required = false) Integer categoryId,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) String priceRange) {
        return ApiResponse.success(productService.listProducts(categoryId, keyword, priceRange));
    }

    @Operation(summary = "Upload product image", description = "Uploads an image for a specific product")
    @PostMapping(path = "/product/{id}/images", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public boolean uploadImage(@PathVariable(name = "id") Long productId, @RequestParam MultipartFile file) {
        return productService.uploadImage(productId, file);
    }
}