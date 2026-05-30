package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.CostDriver;
import java.util.List;

public interface CostDriverRepository {
    CostDriver save(CostDriver costDriver);
    CostDriver findById(Long id);
    CostDriver findByCode(String driverCode);
    List<CostDriver> findAll();
    List<CostDriver> findByType(CostDriver.CostDriverType type);
    void delete(Long id);
}