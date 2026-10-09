package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.ShareRewardConfigDO;
import com.metawebthree.socialcommerce.domain.repository.ShareRewardConfigRepository;

/**
 * 分享奖励配置仓储（本地开发用内存实现）。
 */
@Repository
public class ShareRewardConfigRepositoryImpl implements ShareRewardConfigRepository {

    private final Map<Long, ShareRewardConfigDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(ShareRewardConfigDO config) {
        if (config.getId() == null) {
            config.setId(IdWorker.getId());
        }
        store.put(config.getId(), config);
    }

    @Override
    public void update(ShareRewardConfigDO config) {
        if (config.getId() != null) {
            store.put(config.getId(), config);
        }
    }

    @Override
    public ShareRewardConfigDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public ShareRewardConfigDO findActiveConfig() {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getStatus(), 1))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<ShareRewardConfigDO> findByStatus(Integer status) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
