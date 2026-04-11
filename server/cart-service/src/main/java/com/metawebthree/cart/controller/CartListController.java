package com.metawebthree.cart.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

@RestController
@RequestMapping("/v1/cart")
public class CartListController {

    @GetMapping("/list")
    public ResponseEntity<List<Map<String, Object>>> getCartList(@RequestHeader("X-User-ID") Long userId) {
        List<Map<String, Object>> items = Arrays.asList(
                Map.of("productId", 1L, "name", "Product1", "quantity", 2, "price", 99.99),
                Map.of("productId", 2L, "name", "Product2", "quantity", 1, "price", 199.99));
        return ResponseEntity.ok(items);
    }
}
