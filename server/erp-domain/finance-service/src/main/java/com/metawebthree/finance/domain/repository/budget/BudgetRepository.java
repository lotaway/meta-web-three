package com.metawebthree.finance.domain.repository.budget;

import com.metawebthree.finance.domain.entity.budget.Budget;
import com.metawebthree.finance.domain.entity.budget.BudgetAdjustment;
import com.metawebthree.finance.domain.entity.budget.BudgetLine;

import java.util.List;
import java.util.Optional;

public interface BudgetRepository {
    Optional<Budget> findById(Long id);
    Optional<Budget> findByCode(String budgetCode);
    List<Budget> findByDepartmentId(Long departmentId);
    List<Budget> findByStatus(Budget.BudgetStatus status);
    List<Budget> findByPeriod(Budget.BudgetPeriod period);
    Long save(Budget budget);
    void update(Budget budget);
    void delete(Long id);
    
    List<BudgetLine> findLinesByBudgetId(Long budgetId);
    Long saveLine(BudgetLine line);
    void updateLine(BudgetLine line);
    void deleteLine(Long id);
    
    List<BudgetAdjustment> findAdjustmentsByBudgetId(Long budgetId);
    List<BudgetAdjustment> findPendingAdjustments();
    Long saveAdjustment(BudgetAdjustment adjustment);
    void updateAdjustment(BudgetAdjustment adjustment);
}