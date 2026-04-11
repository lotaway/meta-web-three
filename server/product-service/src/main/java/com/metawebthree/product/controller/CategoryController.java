package com.metawebthree.product.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.metawebthree.product.service.ProductCategoryService;
import com.metawebthree.product.domain.ProductCategory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
@RequestMapping("/v1/product")
public class CategoryController {

    private final ProductCategoryService categoryService;

    public CategoryController(ProductCategoryService categoryService) {
        this.categoryService = categoryService;
    }

    @GetMapping("/category/list")
    public ResponseEntity<List<ProductCategory>> fetchProductCategoryList() {
        return ResponseEntity.ok(categoryService.listCategories());
    }
}
