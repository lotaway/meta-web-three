package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.CostVariance;
import com.metawebthree.finance.domain.repository.cost.CostVarianceRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 成本差异仓储（本地开发用内存实现）。
 */
@Repository
public class CostVarianceRepositoryImpl implements CostVarianceRepository {

    private final Map<Long, CostVariance> store = new ConcurrentHashMap<>();

    @Override
    public CostVariance save(CostVariance costVariance) {
        if (costVariance.getId() == null) {
            costVariance.setId(IdWorker.getId());
        }
        store.put(costVariance.getId(), costVariance);
        return costVariance;
    }

    @Override
    public CostVariance findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<CostVariance> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CostVariance> findByProductCode(String productCode) {
        return store.values().stream()
                .filter(v -> Objects.equals(v.getProductCode(), productCode))
                .collect(Collectors.toList());
    }

    @Override
    public List<CostVariance> findByVarianceDateBetween(LocalDate startDate, LocalDate endDate) {
        return store.values().stream()
                .filter(v -> v.getVarianceDate() != null
                        && (startDate == null || !v.getVarianceDate().isBefore(startDate))
                        && (endDate == null || !v.getVarianceDate().isAfter(endDate)))
                .collect(Collectors.toList());
    }

    @Override
    public List<CostVariance> findByVarianceType(String varianceType) {
        return store.values().stream()
                .filter(v -> Objects.equals(v.getVarianceType(), varianceType))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
