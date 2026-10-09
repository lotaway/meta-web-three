package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.CostDriver;
import com.metawebthree.finance.domain.repository.cost.CostDriverRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 成本动因仓储（本地开发用内存实现）。
 */
@Repository
public class CostDriverRepositoryImpl implements CostDriverRepository {

    private final Map<Long, CostDriver> store = new ConcurrentHashMap<>();

    @Override
    public CostDriver save(CostDriver costDriver) {
        if (costDriver.getId() == null) {
            costDriver.setId(IdWorker.getId());
        }
        store.put(costDriver.getId(), costDriver);
        return costDriver;
    }

    @Override
    public CostDriver findById(Long id) {
        return store.get(id);
    }

    @Override
    public CostDriver findByCode(String driverCode) {
        return store.values().stream()
                .filter(d -> Objects.equals(d.getDriverCode(), driverCode))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<CostDriver> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CostDriver> findByType(CostDriver.CostDriverType type) {
        return store.values().stream()
                .filter(d -> Objects.equals(d.getType(), type))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
