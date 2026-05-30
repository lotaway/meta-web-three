package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.ActualCost;
import java.time.LocalDate;
import java.util.List;

public interface ActualCostRepository {
    ActualCost save(ActualCost actualCost);
    ActualCost findById(Long id);
    List<ActualCost> findAll();
    List<ActualCost> findByProductCode(String productCode);
    List<ActualCost> findByCostCenterId(Long costCenterId);
    List<ActualCost> findByCostDateBetween(LocalDate startDate, LocalDate endDate);
    List<ActualCost> findByProductionOrderNo(String productionOrderNo);
    void delete(Long id);
}