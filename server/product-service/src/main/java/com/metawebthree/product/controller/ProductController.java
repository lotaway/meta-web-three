package com.metawebthree.product.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.service.ProductQueryService;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/v1/product")
public class ProductController {

    private final ProductQueryService productQueryService;

    public ProductController(ProductQueryService productQueryService) {
        this.productQueryService = productQueryService;
    }

    @GetMapping("/{id}")
    public ApiResponse<Map<String, Object>> getProductDetail(@PathVariable Long id,
            @RequestHeader(value = "X-User-ID", required = false) Long userId) {
        return productQueryService.getProductDetail(id, userId);
    }
}
