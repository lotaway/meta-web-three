package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.DistributionRewardDO;
import com.metawebthree.socialcommerce.domain.repository.DistributionRewardRepository;

/**
 * 分销奖励仓储（本地开发用内存实现）。
 */
@Repository
public class DistributionRewardRepositoryImpl implements DistributionRewardRepository {

    private final Map<Long, DistributionRewardDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(DistributionRewardDO reward) {
        if (reward.getId() == null) {
            reward.setId(IdWorker.getId());
        }
        store.put(reward.getId(), reward);
    }

    @Override
    public void update(DistributionRewardDO reward) {
        if (reward.getId() != null) {
            store.put(reward.getId(), reward);
        }
    }

    @Override
    public DistributionRewardDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<DistributionRewardDO> findByReferrerId(Long referrerId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getReferrerId(), referrerId))
                .collect(Collectors.toList());
    }

    @Override
    public List<DistributionRewardDO> findByOrderId(Long orderId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getOrderId(), orderId))
                .collect(Collectors.toList());
    }

    @Override
    public List<DistributionRewardDO> findByStatus(String status) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
