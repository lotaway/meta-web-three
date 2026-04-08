package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.application.BrandApplicationService;
import com.metawebthree.product.application.ProductService;
import com.metawebthree.product.domain.model.Brand;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/brands")
@RequiredArgsConstructor
@Tag(name = "Product Brand Management")
public class BrandController {

    private final BrandApplicationService brandService;
    private final ProductService productService;

    @Operation(summary = "Register brand")
    @PostMapping
    public ApiResponse<Void> register(@RequestBody Brand brand) {
        brandService.registerBrand(brand);
        return ApiResponse.success();
    }

    @Operation(summary = "Get brand details")
    @GetMapping("/{id}")
    public ApiResponse<Brand> details(@PathVariable Long id) {
        return ApiResponse.success(brandService.getBrand(id));
    }

    @Operation(summary = "List brands")
    @GetMapping
    public ApiResponse<List<Brand>> list() {
        return ApiResponse.success(brandService.listBrands());
    }

    @Operation(summary = "Modify brand")
    @PutMapping("/{id}")
    public ApiResponse<Void> modify(@PathVariable Long id, @RequestBody Brand brand) {
        brandService.modifyBrand(brand);
        return ApiResponse.success();
    }

    @Operation(summary = "Remove brand")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        brandService.removeBrand(id);
        return ApiResponse.success();
    }

    @Operation(summary = "获取品牌相关商品列表")
    @GetMapping("/{id}/products")
    public ApiResponse<List<ProductDTO>> listProducts(
            @PathVariable Long id,
            @RequestParam(required = false) String keyword) {
        // 使用现有的 productService.listProducts，传入 brandId 进行过滤
        // 这里需要扩展 listProducts 支持 brandId
        return ApiResponse.success(productService.listProducts(null, keyword, null));
    }
}
