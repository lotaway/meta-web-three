package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.StandardCost;
import com.metawebthree.finance.domain.repository.cost.StandardCostRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 标准成本仓储（本地开发用内存实现）。
 */
@Repository
public class StandardCostRepositoryImpl implements StandardCostRepository {

    private final Map<Long, StandardCost> store = new ConcurrentHashMap<>();

    @Override
    public StandardCost save(StandardCost standardCost) {
        if (standardCost.getId() == null) {
            standardCost.setId(IdWorker.getId());
        }
        store.put(standardCost.getId(), standardCost);
        return standardCost;
    }

    @Override
    public StandardCost findById(Long id) {
        return store.get(id);
    }

    @Override
    public StandardCost findByProductCodeAndEffectiveDate(String productCode, LocalDate effectiveDate) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getProductCode(), productCode)
                        && Objects.equals(c.getEffectiveDate(), effectiveDate))
                .findFirst()
                .orElse(null);
    }

    @Override
    public StandardCost findEffectiveByProductCode(String productCode) {
        LocalDate today = LocalDate.now();
        return store.values().stream()
                .filter(c -> Objects.equals(c.getProductCode(), productCode))
                .filter(c -> c.getEffectiveDate() == null || !c.getEffectiveDate().isAfter(today))
                .max(Comparator.comparing(StandardCost::getEffectiveDate,
                        Comparator.nullsFirst(Comparator.naturalOrder())))
                .orElse(null);
    }

    @Override
    public List<StandardCost> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<StandardCost> findByProductCategory(String category) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getProductCategory(), category))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
