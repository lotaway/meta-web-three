package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.ResourcePool;
import com.metawebthree.finance.domain.repository.cost.ResourcePoolRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 资源池仓储（本地开发用内存实现）。
 */
@Repository
public class ResourcePoolRepositoryImpl implements ResourcePoolRepository {

    private final Map<Long, ResourcePool> store = new ConcurrentHashMap<>();

    @Override
    public ResourcePool save(ResourcePool resourcePool) {
        if (resourcePool.getId() == null) {
            resourcePool.setId(IdWorker.getId());
        }
        store.put(resourcePool.getId(), resourcePool);
        return resourcePool;
    }

    @Override
    public ResourcePool findById(Long id) {
        return store.get(id);
    }

    @Override
    public ResourcePool findByCode(String poolCode) {
        return store.values().stream()
                .filter(p -> Objects.equals(p.getPoolCode(), poolCode))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<ResourcePool> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<ResourcePool> findByCostCenterId(Long costCenterId) {
        return store.values().stream()
                .filter(p -> Objects.equals(p.getCostCenterId(), costCenterId))
                .collect(Collectors.toList());
    }

    @Override
    public List<ResourcePool> findByType(ResourcePool.ResourcePoolType type) {
        return store.values().stream()
                .filter(p -> Objects.equals(p.getType(), type))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
