package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.CostVariance;
import java.time.LocalDate;
import java.util.List;

public interface CostVarianceRepository {
    CostVariance save(CostVariance costVariance);
    CostVariance findById(Long id);
    List<CostVariance> findAll();
    List<CostVariance> findByProductCode(String productCode);
    List<CostVariance> findByVarianceDateBetween(LocalDate startDate, LocalDate endDate);
    List<CostVariance> findByVarianceType(String varianceType);
    void delete(Long id);
}