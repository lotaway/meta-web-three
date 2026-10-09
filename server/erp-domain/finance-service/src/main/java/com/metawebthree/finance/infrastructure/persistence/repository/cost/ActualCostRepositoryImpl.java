package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.ActualCost;
import com.metawebthree.finance.domain.repository.cost.ActualCostRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 实际成本仓储（本地开发用内存实现）。
 */
@Repository
public class ActualCostRepositoryImpl implements ActualCostRepository {

    private final Map<Long, ActualCost> store = new ConcurrentHashMap<>();

    @Override
    public ActualCost save(ActualCost actualCost) {
        if (actualCost.getId() == null) {
            actualCost.setId(IdWorker.getId());
        }
        store.put(actualCost.getId(), actualCost);
        return actualCost;
    }

    @Override
    public ActualCost findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<ActualCost> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<ActualCost> findByProductCode(String productCode) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getProductCode(), productCode))
                .collect(Collectors.toList());
    }

    @Override
    public List<ActualCost> findByCostCenterId(Long costCenterId) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getCostCenterId(), costCenterId))
                .collect(Collectors.toList());
    }

    @Override
    public List<ActualCost> findByCostDateBetween(LocalDate startDate, LocalDate endDate) {
        return store.values().stream()
                .filter(c -> c.getCostDate() != null
                        && (startDate == null || !c.getCostDate().isBefore(startDate))
                        && (endDate == null || !c.getCostDate().isAfter(endDate)))
                .collect(Collectors.toList());
    }

    @Override
    public List<ActualCost> findByProductionOrderNo(String productionOrderNo) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getProductionOrderNo(), productionOrderNo))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
