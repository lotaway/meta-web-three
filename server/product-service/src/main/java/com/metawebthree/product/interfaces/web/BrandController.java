package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.application.BrandApplicationService;
import com.metawebthree.product.application.ProductService;
import com.metawebthree.product.domain.model.Brand;
import com.metawebthree.product.domain.model.BrandDO;
import com.metawebthree.product.infrastructure.persistence.mapper.BrandMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@RestController
@RequestMapping("/brand")
@RequiredArgsConstructor
@Tag(name = "Product Brand Management")
public class BrandController {

    private final BrandApplicationService brandService;
    private final ProductService productService;
    private final BrandMapper brandMapper;

    @Operation(summary = "Register brand")
    @PostMapping
    public ApiResponse<Void> register(@RequestBody Brand brand) {
        brandService.registerBrand(brand);
        return ApiResponse.success();
    }

    @Operation(summary = "添加品牌")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody BrandDO brand) {
        brandMapper.insert(brand);
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

    @Operation(summary = "更新品牌")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> updateById(@PathVariable Long id, @RequestBody BrandDO brand) {
        brand.setId(id);
        brandMapper.updateById(brand);
        return ApiResponse.success();
    }

    @Operation(summary = "删除品牌")
    @GetMapping("/delete/{id}")
    public ApiResponse<Void> deleteById(@PathVariable Long id) {
        brandService.removeBrand(id);
        return ApiResponse.success();
    }

    @Operation(summary = "分页获取品牌列表")
    @GetMapping("/list")
    public ApiResponse<Page<BrandDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(brandMapper.selectPage(new Page<>(pageNum, pageSize), null));
    }

    @Operation(summary = "获取全部品牌列表")
    @GetMapping("/listAll")
    public ApiResponse<List<BrandDO>> listAll() {
        return ApiResponse.success(brandMapper.selectList(null));
    }

    @Operation(summary = "批量更新显示状态")
    @PostMapping("/update/showStatus")
    public ApiResponse<Void> updateShowStatus(
            @RequestParam String ids,
            @RequestParam Integer showStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        brandMapper.update(null, new UpdateWrapper<BrandDO>()
                .in("id", idList).set("show_status", showStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量更新厂家制造商状态")
    @PostMapping("/update/factoryStatus")
    public ApiResponse<Void> updateFactoryStatus(
            @RequestParam String ids,
            @RequestParam Integer factoryStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        brandMapper.update(null, new UpdateWrapper<BrandDO>()
                .in("id", idList).set("factory_status", factoryStatus));
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
