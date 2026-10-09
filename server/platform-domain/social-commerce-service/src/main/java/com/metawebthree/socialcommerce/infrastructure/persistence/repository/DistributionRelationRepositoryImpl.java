package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.DistributionRelationDO;
import com.metawebthree.socialcommerce.domain.repository.DistributionRelationRepository;

/**
 * 分销关系仓储（本地开发用内存实现）。
 */
@Repository
public class DistributionRelationRepositoryImpl implements DistributionRelationRepository {

    private final Map<Long, DistributionRelationDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(DistributionRelationDO relation) {
        if (relation.getId() == null) {
            relation.setId(IdWorker.getId());
        }
        store.put(relation.getId(), relation);
    }

    @Override
    public void update(DistributionRelationDO relation) {
        if (relation.getId() != null) {
            store.put(relation.getId(), relation);
        }
    }

    @Override
    public DistributionRelationDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public DistributionRelationDO findByUserId(Long userId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getUserId(), userId))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<DistributionRelationDO> findByReferrerId(Long referrerId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getReferrerId(), referrerId))
                .collect(Collectors.toList());
    }

    @Override
    public List<DistributionRelationDO> findByRootReferrerId(Long rootReferrerId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getRootReferrerId(), rootReferrerId))
                .collect(Collectors.toList());
    }

    @Override
    public Boolean existsByUserId(Long userId) {
        return store.values().stream().anyMatch(r -> Objects.equals(r.getUserId(), userId));
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
