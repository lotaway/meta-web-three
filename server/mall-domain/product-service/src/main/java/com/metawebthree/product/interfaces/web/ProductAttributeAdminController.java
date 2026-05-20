package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.AttributeDO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductAttributeMapper;
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
@RequestMapping("/productAttribute")
@RequiredArgsConstructor
@Tag(name = "Product Attribute Admin")
public class ProductAttributeAdminController {

    private final ProductAttributeMapper attributeMapper;

    @Operation(summary = "根据分类ID查询属性列表或参数列表")
    @GetMapping("/list/{cid}")
    public ApiResponse<Page<AttributeDO>> list(
            @PathVariable Long cid,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam Integer type) {
        LambdaQueryWrapper<AttributeDO> wrapper = new LambdaQueryWrapper<AttributeDO>()
                .eq(AttributeDO::getProductAttributeCategoryId, cid)
                .eq(AttributeDO::getType, type)
                .orderByDesc(AttributeDO::getId);
        return ApiResponse.success(attributeMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "根据商品分类的ID获取商品属性及属性分类ID")
    @GetMapping("/attrInfo/{cateId}")
    public ApiResponse<List<AttributeDO>> attrInfo(@PathVariable Long cateId) {
        LambdaQueryWrapper<AttributeDO> wrapper = new LambdaQueryWrapper<AttributeDO>()
                .eq(AttributeDO::getProductAttributeCategoryId, cateId);
        return ApiResponse.success(attributeMapper.selectList(wrapper));
    }

    @Operation(summary = "添加商品属性信息")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody AttributeDO attribute) {
        attributeMapper.insert(attribute);
        return ApiResponse.success();
    }

    @Operation(summary = "修改商品属性信息")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody AttributeDO attribute) {
        attribute.setId(id);
        attributeMapper.updateById(attribute);
        return ApiResponse.success();
    }

    @Operation(summary = "根据ID查询商品属性")
    @GetMapping("/{id}")
    public ApiResponse<AttributeDO> getById(@PathVariable Long id) {
        return ApiResponse.success(attributeMapper.selectById(id));
    }

    @Operation(summary = "批量删除商品属性")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        attributeMapper.deleteByIds(idList);
        return ApiResponse.success();
    }
}
