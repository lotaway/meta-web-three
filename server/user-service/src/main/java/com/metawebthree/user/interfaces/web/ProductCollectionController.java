package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@RestController
@RequestMapping("/member/productCollection")
public class ProductCollectionController {

    private final Map<Long, List<ProductCollectionItem>> collectionStore = new ConcurrentHashMap<>();

    @PostMapping("/add")
    public ApiResponse<Void> add(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                  @RequestBody ProductCollectionItem item) {
        item.setUserId(userId);
        item.setCreateTime(System.currentTimeMillis());
        
        List<ProductCollectionItem> collections = collectionStore.computeIfAbsent(userId, k -> new ArrayList<>());
        
        // 检查是否已收藏
        boolean exists = collections.stream().anyMatch(c -> c.getProductId().equals(item.getProductId()));
        if (!exists) {
            collections.add(0, item);
        }
        
        return ApiResponse.success();
    }

    @GetMapping("/list")
    public ApiResponse<List<ProductCollectionItem>> list(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                           @RequestParam(defaultValue = "1") Integer pageNum,
                                                           @RequestParam(defaultValue = "20") Integer pageSize) {
        List<ProductCollectionItem> collections = collectionStore.getOrDefault(userId, new ArrayList<>());
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, collections.size());
        
        if (start >= collections.size()) {
            return ApiResponse.success(new ArrayList<>());
        }
        
        return ApiResponse.success(collections.subList(start, end));
    }

    @DeleteMapping
    public ApiResponse<Void> delete(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                     @RequestParam Long productId) {
        List<ProductCollectionItem> collections = collectionStore.get(userId);
        if (collections != null) {
            collections.removeIf(item -> item.getProductId().equals(productId));
        }
        return ApiResponse.success();
    }

    @GetMapping("/detail")
    public ApiResponse<ProductCollectionItem> detail(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                       @RequestParam Long productId) {
        List<ProductCollectionItem> collections = collectionStore.get(userId);
        if (collections != null) {
            return collections.stream()
                    .filter(item -> item.getProductId().equals(productId))
                    .findFirst()
                    .map(ApiResponse::success)
                    .orElse(ApiResponse.success(null));
        }
        return ApiResponse.success(null);
    }

    @DeleteMapping("/clear")
    public ApiResponse<Void> clear(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        collectionStore.remove(userId);
        return ApiResponse.success();
    }

    @Data
    public static class ProductCollectionItem {
        private Long userId;
        private Long productId;
        private String productName;
        private String productPic;
        private Long createTime;
    }
}
