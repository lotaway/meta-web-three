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
@RequestMapping("/brand")
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
    @GetMapping({"/{id}", "/detail/{id}"})
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
        return ApiResponse.success(productService.listProducts(null, keyword, null));
    }

    @Operation(summary = "获取品牌相关商品列表（别名）")
    @GetMapping("/productList")
    public ApiResponse<List<ProductDTO>> listProductsByBrand(
            @RequestParam(required = false) Long brandId,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.success(productService.listProducts(null, keyword, null));
    }

    @Operation(summary = "获取推荐品牌列表")
    @GetMapping("/recommendList")
    public ApiResponse<List<Brand>> recommendList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        // 返回所有品牌作为推荐（简化实现）
        return ApiResponse.success(brandService.listBrands());
    }
}
