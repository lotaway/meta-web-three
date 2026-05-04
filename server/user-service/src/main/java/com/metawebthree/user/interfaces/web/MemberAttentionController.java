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
@RequestMapping("/member/attention")
public class MemberAttentionController {

    private final Map<Long, List<BrandAttentionItem>> attentionStore = new ConcurrentHashMap<>();

    @PostMapping("/add")
    public ApiResponse<Void> add(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                  @RequestBody BrandAttentionItem item) {
        item.setUserId(userId);
        item.setCreateTime(System.currentTimeMillis());
        
        List<BrandAttentionItem> attentions = attentionStore.computeIfAbsent(userId, k -> new ArrayList<>());
        
        // 检查是否已关注
        boolean exists = attentions.stream().anyMatch(a -> a.getBrandId().equals(item.getBrandId()));
        if (!exists) {
            attentions.add(0, item);
        }
        
        return ApiResponse.success();
    }

    @GetMapping("/list")
    public ApiResponse<List<BrandAttentionItem>> list(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                        @RequestParam(defaultValue = "1") Integer pageNum,
                                                        @RequestParam(defaultValue = "20") Integer pageSize) {
        List<BrandAttentionItem> attentions = attentionStore.getOrDefault(userId, new ArrayList<>());
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, attentions.size());
        
        if (start >= attentions.size()) {
            return ApiResponse.success(new ArrayList<>());
        }
        
        return ApiResponse.success(attentions.subList(start, end));
    }

    @DeleteMapping
    public ApiResponse<Void> delete(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                     @RequestParam Long brandId) {
        List<BrandAttentionItem> attentions = attentionStore.get(userId);
        if (attentions != null) {
            attentions.removeIf(item -> item.getBrandId().equals(brandId));
        }
        return ApiResponse.success();
    }

    @GetMapping("/detail")
    public ApiResponse<BrandAttentionItem> detail(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                    @RequestParam Long brandId) {
        List<BrandAttentionItem> attentions = attentionStore.get(userId);
        if (attentions != null) {
            return attentions.stream()
                    .filter(item -> item.getBrandId().equals(brandId))
                    .findFirst()
                    .map(ApiResponse::success)
                    .orElse(ApiResponse.success(null));
        }
        return ApiResponse.success(null);
    }

    @DeleteMapping("/clear")
    public ApiResponse<Void> clear(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        attentionStore.remove(userId);
        return ApiResponse.success();
    }

    @Data
    public static class BrandAttentionItem {
        private Long userId;
        private Long brandId;
        private String brandName;
        private String brandLogo;
        private Long createTime;
    }
}
