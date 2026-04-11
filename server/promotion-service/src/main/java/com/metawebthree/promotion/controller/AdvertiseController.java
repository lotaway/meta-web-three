package com.metawebthree.promotion.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;
import com.metawebthree.promotion.application.AdvertiseService;
import com.metawebthree.promotion.domain.model.Advertise;
import lombok.RequiredArgsConstructor;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/v1/promotion")
@RequiredArgsConstructor
public class AdvertiseController {
    private final AdvertiseService advertiseService;

    @GetMapping("/advertises")
    public ResponseEntity<List<Map<String, Object>>> getAdvertises() {
        List<Advertise> advertises = advertiseService.listAvailable(1);
        List<Map<String, Object>> result = advertises.stream()
                .map(ad -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("id", ad.getId());
                    map.put("pic", ad.getPic());
                    map.put("url", ad.getUrl());
                    return map;
                })
                .collect(Collectors.toList());
        return ResponseEntity.ok(result);
    }
}
