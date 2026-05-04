package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@RestController
@RequestMapping("/member/readHistory")
public class ReadHistoryController {

    private final Map<Long, List<ReadHistoryItem>> historyStore = new ConcurrentHashMap<>();

    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                     @RequestBody ReadHistoryItem item) {
        item.setUserId(userId);
        item.setCreateTime(System.currentTimeMillis());
        historyStore.computeIfAbsent(userId, k -> new ArrayList<>()).add(0, item);
        
        // 限制最多 50 条
        List<ReadHistoryItem> history = historyStore.get(userId);
        if (history.size() > 50) {
            history.subList(50, history.size()).clear();
        }
        
        return ApiResponse.success();
    }

    @GetMapping("/list")
    public ApiResponse<List<ReadHistoryItem>> list(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                     @RequestParam(defaultValue = "1") Integer pageNum,
                                                     @RequestParam(defaultValue = "20") Integer pageSize) {
        List<ReadHistoryItem> history = historyStore.getOrDefault(userId, new ArrayList<>());
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, history.size());
        
        if (start >= history.size()) {
            return ApiResponse.success(new ArrayList<>());
        }
        
        return ApiResponse.success(history.subList(start, end));
    }

    @DeleteMapping
    public ApiResponse<Void> delete(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                     @RequestParam Long productId) {
        List<ReadHistoryItem> history = historyStore.get(userId);
        if (history != null) {
            history.removeIf(item -> item.getProductId().equals(productId));
        }
        return ApiResponse.success();
    }

    @DeleteMapping("/clear")
    public ApiResponse<Void> clear(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        historyStore.remove(userId);
        return ApiResponse.success();
    }

    @Data
    public static class ReadHistoryItem {
        private Long userId;
        private Long productId;
        private String productName;
        private String productPic;
        private Long createTime;
    }
}
