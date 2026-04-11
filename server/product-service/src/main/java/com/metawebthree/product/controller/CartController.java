package com.metawebthree.product.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;
import java.util.Map;

@RestController
@RequestMapping("/v1/cart")
public class CartController {

    @PostMapping("/add")
    public ResponseEntity<Map<String, Object>> addToCart(@RequestBody Map<String, Object> request,
            @RequestHeader("X-User-ID") Long userId) {
        Map<String, Object> result = Map.of("success", true, "message", "Added to cart");
        return ResponseEntity.ok(result);
    }
}
