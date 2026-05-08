package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.ProductAttributeCategoryDO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductAttributeCategoryMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/productAttribute/category")
@RequiredArgsConstructor
@Tag(name = "Product Attribute Category Admin")
public class ProductAttributeCategoryAdminController {

    private final ProductAttributeCategoryMapper attributeCategoryMapper;

    @Operation(summary = "分页获取所有商品属性分类")
    @GetMapping("/list")
    public ApiResponse<Page<ProductAttributeCategoryDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(attributeCategoryMapper.selectPage(new Page<>(pageNum, pageSize), null));
    }

    @Operation(summary = "获取所有商品属性分类及其下属性")
    @GetMapping("/list/withAttr")
    public ApiResponse<Page<ProductAttributeCategoryDO>> listWithAttr() {
        return ApiResponse.success(attributeCategoryMapper.selectPage(new Page<>(1, 100), null));
    }

    @Operation(summary = "添加商品属性分类")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestParam String name) {
        ProductAttributeCategoryDO entity = new ProductAttributeCategoryDO();
        entity.setName(name);
        attributeCategoryMapper.insert(entity);
        return ApiResponse.success();
    }

    @Operation(summary = "修改商品属性分类")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestParam String name) {
        ProductAttributeCategoryDO entity = attributeCategoryMapper.selectById(id);
        if (entity != null) {
            entity.setName(name);
            attributeCategoryMapper.updateById(entity);
        }
        return ApiResponse.success();
    }

    @Operation(summary = "删除单个商品属性分类")
    @GetMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        attributeCategoryMapper.deleteById(id);
        return ApiResponse.success();
    }
}
