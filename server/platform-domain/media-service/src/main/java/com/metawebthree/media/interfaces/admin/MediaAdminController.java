package com.metawebthree.media.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.media.config.UploadQuotaProperties;
import com.metawebthree.media.domain.model.UserStorageDO;
import com.metawebthree.media.infrastructure.persistence.mapper.UserStorageMapper;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/media")
@RequiredArgsConstructor
public class MediaAdminController {

    private final UserStorageMapper userStorageMapper;
    private final UploadQuotaProperties uploadQuotaProperties;

    @GetMapping("/storage/list")
    public ApiResponse<Map<String, Object>> listUserStorage(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId) {

        LambdaQueryWrapper<UserStorageDO> query = new LambdaQueryWrapper<UserStorageDO>();

        if (userId != null) {
            query.eq(UserStorageDO::getUserId, userId);
        }

        query.orderByDesc(UserStorageDO::getUpdatedAt);

        Page<UserStorageDO> page = new Page<>(pageNum, pageSize);
        Page<UserStorageDO> result = userStorageMapper.selectPage(page, query);

        Map<String, Object> response = new HashMap<>();
        response.put("list", result.getRecords());
        response.put("total", result.getTotal());
        response.put("pageNum", result.getCurrent());
        response.put("pageSize", result.getSize());

        return ApiResponse.success(response);
    }

    @GetMapping("/storage/{id}")
    public ApiResponse<UserStorageDO> getStorageById(@PathVariable Long id) {
        UserStorageDO storage = userStorageMapper.selectById(id);
        if (storage != null) {
            return ApiResponse.success(storage);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "User storage record not found");
    }

    @GetMapping("/storage/user/{userId}")
    public ApiResponse<UserStorageDO> getStorageByUserId(@PathVariable Long userId) {
        UserStorageDO storage = userStorageMapper.selectOne(
                new LambdaQueryWrapper<UserStorageDO>().eq(UserStorageDO::getUserId, userId));
        if (storage != null) {
            return ApiResponse.success(storage);
        }
        return ApiResponse.success(null);
    }

    @DeleteMapping("/storage/{id}")
    public ApiResponse<Void> deleteStorage(@PathVariable Long id) {
        int result = userStorageMapper.deleteById(id);
        if (result > 0) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "User storage record not found");
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        Long totalUsers = userStorageMapper.selectCount(null);
        
        // Calculate total storage used
        java.util.List<UserStorageDO> allStorages = userStorageMapper.selectList(null);
        long totalUsed = 0L;
        long maxUsed = 0L;
        Long maxUsedUserId = null;
        
        for (UserStorageDO storage : allStorages) {
            totalUsed += storage.getTotalUsed() != null ? storage.getTotalUsed() : 0L;
            if (storage.getTotalUsed() != null && storage.getTotalUsed() > maxUsed) {
                maxUsed = storage.getTotalUsed();
                maxUsedUserId = storage.getUserId();
            }
        }

        Map<String, Object> statistics = new HashMap<>();
        statistics.put("totalUsers", totalUsers);
        statistics.put("totalUsed", totalUsed);
        statistics.put("maxUsed", maxUsed);
        statistics.put("maxUsedUserId", maxUsedUserId);
        statistics.put("averageUsed", totalUsers > 0 ? totalUsed / totalUsers : 0);

        return ApiResponse.success(statistics);
    }

    @GetMapping("/quota/config")
    public ApiResponse<Map<String, Object>> getQuotaConfig() {
        Map<String, Object> config = new HashMap<>();
        Map<String, UploadQuotaProperties.QuotaConfig> quotas = uploadQuotaProperties.getQuotas();
        
        Map<String, Map<String, Long>> quotaMap = new HashMap<>();
        for (Map.Entry<String, UploadQuotaProperties.QuotaConfig> entry : quotas.entrySet()) {
            Map<String, Long> quotaInfo = new HashMap<>();
            quotaInfo.put("maxFileSize", entry.getValue().getMaxFileSize());
            quotaInfo.put("totalQuota", entry.getValue().getTotalQuota());
            quotaMap.put(entry.getKey(), quotaInfo);
        }
        
        config.put("quotas", quotaMap);
        return ApiResponse.success(config);
    }
}