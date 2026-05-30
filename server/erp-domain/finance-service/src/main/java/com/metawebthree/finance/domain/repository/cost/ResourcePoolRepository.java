package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.ResourcePool;
import java.util.List;

public interface ResourcePoolRepository {
    ResourcePool save(ResourcePool resourcePool);
    ResourcePool findById(Long id);
    ResourcePool findByCode(String poolCode);
    List<ResourcePool> findAll();
    List<ResourcePool> findByCostCenterId(Long costCenterId);
    List<ResourcePool> findByType(ResourcePool.ResourcePoolType type);
    void delete(Long id);
}