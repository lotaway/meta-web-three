package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.application.ProductCategoryApplicationService;
import com.metawebthree.product.domain.model.ProductCategory;
import com.metawebthree.product.interfaces.web.dto.ProductCategoryNode;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/product-categories")
@RequiredArgsConstructor
@Tag(name = "Product Category Management")
public class ProductCategoryController {

    private final ProductCategoryApplicationService categoryService;

    @Operation(summary = "Create category")
    @PostMapping
    public ApiResponse<Void> create(@RequestBody ProductCategory category) {
        categoryService.createCategory(category);
        return ApiResponse.success();
    }

    @Operation(summary = "View sub-categories")
    @GetMapping("/children/{parentId}")
    public ApiResponse<List<ProductCategory>> viewChildren(@PathVariable Long parentId) {
        return ApiResponse.success(categoryService.findSubCategories(parentId));
    }

    @Operation(summary = "以树形结构获取所有商品分类")
    @GetMapping("/tree")
    public ApiResponse<List<ProductCategoryNode>> categoryTreeList() {
        return ApiResponse.success(categoryService.categoryTreeList());
    }

    @Operation(summary = "Update category")
    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody ProductCategory category) {
        categoryService.updateCategory(category);
        return ApiResponse.success();
    }

    @Operation(summary = "Remove category")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        categoryService.removeCategory(id);
        return ApiResponse.success();
    }
}
