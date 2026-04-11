package com.metawebthree.product.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;
import java.util.Map;
import java.util.HashMap;

@RestController
@RequestMapping("/v1/home")
public class HomeController {

    @GetMapping("/content")
    public ResponseEntity<Map<String, Object>> getHomeContent(
            @RequestHeader(value = "X-User-ID", required = false) Long userId) {
        Map<String, Object> data = new HashMap<>();
        data.put("brands", new String[] { "MetaBrand1", "MetaBrand2" });
        data.put("keywords", new String[] { "Hot", "Recommended" });
        data.put("categories", new String[] { "Electronics", "Fashion" });

        return ResponseEntity.ok(data);
    }
}
