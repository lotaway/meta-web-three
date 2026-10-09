package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.CostCenter;
import com.metawebthree.finance.domain.repository.cost.CostCenterRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 成本中心仓储（本地开发用内存实现）。
 */
@Repository
public class CostCenterRepositoryImpl implements CostCenterRepository {

    private final Map<Long, CostCenter> store = new ConcurrentHashMap<>();

    @Override
    public CostCenter save(CostCenter costCenter) {
        if (costCenter.getId() == null) {
            costCenter.setId(IdWorker.getId());
        }
        store.put(costCenter.getId(), costCenter);
        return costCenter;
    }

    @Override
    public CostCenter findById(Long id) {
        return store.get(id);
    }

    @Override
    public CostCenter findByCode(String costCenterCode) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getCostCenterCode(), costCenterCode))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<CostCenter> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CostCenter> findByType(CostCenter.CostCenterType type) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getType(), type))
                .collect(Collectors.toList());
    }

    @Override
    public List<CostCenter> findByDepartmentId(Long departmentId) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getDepartmentId(), departmentId))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
