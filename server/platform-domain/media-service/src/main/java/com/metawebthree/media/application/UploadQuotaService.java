package com.metawebthree.media.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.media.config.UploadQuotaProperties;
import com.metawebthree.media.domain.model.UserStorageDO;
import com.metawebthree.media.infrastructure.persistence.mapper.UserStorageMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UploadQuotaService {

    private final UserStorageMapper userStorageMapper;
    private final UploadQuotaProperties quotaProperties;

    public void checkQuota(String roleName, long fileSize, Long userId) {
        UploadQuotaProperties.QuotaConfig quota = quotaProperties.getQuotas().get(roleName);
        if (quota == null) {
            return;
        }
        if (fileSize > quota.getMaxFileSize()) {
            throw new IllegalArgumentException("文件大小超过单次上传限制");
        }
        if (userId != null) {
            UserStorageDO storage = userStorageMapper.selectOne(
                    new LambdaQueryWrapper<UserStorageDO>().eq(UserStorageDO::getUserId, userId));
            long currentUsed = storage != null ? storage.getTotalUsed() : 0L;
            if (currentUsed + fileSize > quota.getTotalQuota()) {
                throw new IllegalArgumentException("存储配额不足");
            }
        }
    }

    @Transactional
    public void trackUpload(Long userId, long fileSize) {
        if (userId == null) {
            return;
        }
        UserStorageDO storage = userStorageMapper.selectOne(
                new LambdaQueryWrapper<UserStorageDO>().eq(UserStorageDO::getUserId, userId));
        if (storage == null) {
            storage = new UserStorageDO();
            storage.setUserId(userId);
            storage.setTotalUsed(fileSize);
            storage.setCreatedAt(java.time.LocalDateTime.now());
            storage.setUpdatedAt(storage.getCreatedAt());
            userStorageMapper.insert(storage);
        } else {
            storage.setTotalUsed(storage.getTotalUsed() + fileSize);
            storage.setUpdatedAt(java.time.LocalDateTime.now());
            userStorageMapper.updateById(storage);
        }
    }
}
