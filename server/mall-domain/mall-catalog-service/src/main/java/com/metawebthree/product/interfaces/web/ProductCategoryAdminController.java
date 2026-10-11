package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.ProductCategoryDO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductCategoryMapper;
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
@RequestMapping("/productCategory")
@RequiredArgsConstructor
@Tag(name = "Product Category Admin")
public class ProductCategoryAdminController {

    private final ProductCategoryMapper productCategoryMapper;

    @Operation(summary = "分页查询商品分类")
    @GetMapping("/list/{parentId}")
    public ApiResponse<Page<ProductCategoryDO>> list(
            @PathVariable Long parentId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        LambdaQueryWrapper<ProductCategoryDO> wrapper = new LambdaQueryWrapper<ProductCategoryDO>()
                .eq(ProductCategoryDO::getParentId, parentId)
                .orderByDesc(ProductCategoryDO::getId);
        return ApiResponse.success(productCategoryMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "查询所有一级分类及子分类")
    @GetMapping("/list/withChildren")
    public ApiResponse<List<ProductCategoryDO>> listWithChildren() {
        List<ProductCategoryDO> all = productCategoryMapper.selectList(null);
        return ApiResponse.success(all);
    }

    @Operation(summary = "根据ID获取商品分类")
    @GetMapping("/{id}")
    public ApiResponse<ProductCategoryDO> getById(@PathVariable Long id) {
        return ApiResponse.success(productCategoryMapper.selectById(id));
    }

    @Operation(summary = "添加商品分类")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody ProductCategoryDO category) {
        productCategoryMapper.insert(category);
        return ApiResponse.success();
    }

    @Operation(summary = "修改商品分类")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody ProductCategoryDO category) {
        category.setId(id);
        productCategoryMapper.updateById(category);
        return ApiResponse.success();
    }

    @Operation(summary = "删除商品分类")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        productCategoryMapper.deleteById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "批量修改导航栏显示状态")
    @PostMapping("/update/navStatus")
    public ApiResponse<Void> updateNavStatus(
            @RequestParam String ids,
            @RequestParam Integer navStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        productCategoryMapper.update(null, new UpdateWrapper<ProductCategoryDO>()
                .in("id", idList).set("nav_status", navStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量修改显示状态")
    @PostMapping("/update/showStatus")
    public ApiResponse<Void> updateShowStatus(
            @RequestParam String ids,
            @RequestParam Integer showStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        productCategoryMapper.update(null, new UpdateWrapper<ProductCategoryDO>()
                .in("id", idList).set("show_status", showStatus));
        return ApiResponse.success();
    }
}
