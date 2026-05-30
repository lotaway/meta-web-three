package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.CostCenter;
import java.util.List;

public interface CostCenterRepository {
    CostCenter save(CostCenter costCenter);
    CostCenter findById(Long id);
    CostCenter findByCode(String costCenterCode);
    List<CostCenter> findAll();
    List<CostCenter> findByType(CostCenter.CostCenterType type);
    List<CostCenter> findByDepartmentId(Long departmentId);
    void delete(Long id);
}