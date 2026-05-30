package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.StandardCost;
import java.time.LocalDate;
import java.util.List;

public interface StandardCostRepository {
    StandardCost save(StandardCost standardCost);
    StandardCost findById(Long id);
    StandardCost findByProductCodeAndEffectiveDate(String productCode, LocalDate effectiveDate);
    StandardCost findEffectiveByProductCode(String productCode);
    List<StandardCost> findAll();
    List<StandardCost> findByProductCategory(String category);
    void delete(Long id);
}