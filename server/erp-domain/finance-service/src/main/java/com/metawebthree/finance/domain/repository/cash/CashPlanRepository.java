package com.metawebthree.finance.domain.repository.cash;

import com.metawebthree.finance.domain.entity.cash.CashPlan;
import com.metawebthree.finance.domain.entity.cash.CashPlanLine;
import java.util.List;
import java.util.Optional;

public interface CashPlanRepository {
    Long save(CashPlan cashPlan);
    void update(CashPlan cashPlan);
    Optional<CashPlan> findById(Long id);
    Optional<CashPlan> findByPlanCode(String planCode);
    List<CashPlan> findAll();
    List<CashPlan> findByStatus(CashPlan.CashPlanStatus status);
    List<CashPlan> findByDepartmentId(Long departmentId);
    List<CashPlan> findByPeriod(CashPlan.CashPlanPeriod period);
    void deleteById(Long id);
    
    // Line operations
    Long saveLine(CashPlanLine line);
    List<CashPlanLine> findLinesByCashPlanId(Long cashPlanId);
    void deleteLine(Long lineId);
}