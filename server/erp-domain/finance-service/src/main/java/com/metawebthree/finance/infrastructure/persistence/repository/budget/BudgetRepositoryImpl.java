package com.metawebthree.finance.infrastructure.persistence.repository.budget;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.budget.Budget;
import com.metawebthree.finance.domain.entity.budget.BudgetAdjustment;
import com.metawebthree.finance.domain.entity.budget.BudgetLine;
import com.metawebthree.finance.domain.repository.budget.BudgetRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 预算仓储（本地开发用内存实现）。预算相关数据表尚未落地，这里以内存存储保证服务可独立启动。
 */
@Repository
public class BudgetRepositoryImpl implements BudgetRepository {

    private final Map<Long, Budget> budgets = new ConcurrentHashMap<>();
    private final Map<Long, BudgetLine> lines = new ConcurrentHashMap<>();
    private final Map<Long, BudgetAdjustment> adjustments = new ConcurrentHashMap<>();

    @Override
    public Optional<Budget> findById(Long id) {
        return Optional.ofNullable(budgets.get(id));
    }

    @Override
    public Optional<Budget> findByCode(String budgetCode) {
        return budgets.values().stream()
                .filter(b -> Objects.equals(b.getBudgetCode(), budgetCode))
                .findFirst();
    }

    @Override
    public List<Budget> findByDepartmentId(Long departmentId) {
        return budgets.values().stream()
                .filter(b -> Objects.equals(b.getDepartmentId(), departmentId))
                .collect(Collectors.toList());
    }

    @Override
    public List<Budget> findByStatus(Budget.BudgetStatus status) {
        return budgets.values().stream()
                .filter(b -> Objects.equals(b.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<Budget> findByPeriod(Budget.BudgetPeriod period) {
        return budgets.values().stream()
                .filter(b -> Objects.equals(b.getPeriod(), period))
                .collect(Collectors.toList());
    }

    @Override
    public Long save(Budget budget) {
        if (budget.getId() == null) {
            budget.setId(IdWorker.getId());
        }
        budgets.put(budget.getId(), budget);
        return budget.getId();
    }

    @Override
    public void update(Budget budget) {
        if (budget.getId() != null) {
            budgets.put(budget.getId(), budget);
        }
    }

    @Override
    public void delete(Long id) {
        budgets.remove(id);
    }

    @Override
    public List<BudgetLine> findLinesByBudgetId(Long budgetId) {
        return lines.values().stream()
                .filter(l -> Objects.equals(l.getBudgetId(), budgetId))
                .collect(Collectors.toList());
    }

    @Override
    public Long saveLine(BudgetLine line) {
        if (line.getId() == null) {
            line.setId(IdWorker.getId());
        }
        lines.put(line.getId(), line);
        return line.getId();
    }

    @Override
    public void updateLine(BudgetLine line) {
        if (line.getId() != null) {
            lines.put(line.getId(), line);
        }
    }

    @Override
    public void deleteLine(Long id) {
        lines.remove(id);
    }

    @Override
    public List<BudgetAdjustment> findAdjustmentsByBudgetId(Long budgetId) {
        return adjustments.values().stream()
                .filter(a -> Objects.equals(a.getBudgetId(), budgetId))
                .collect(Collectors.toList());
    }

    @Override
    public List<BudgetAdjustment> findPendingAdjustments() {
        return adjustments.values().stream()
                .filter(a -> Objects.equals(a.getStatus(), BudgetAdjustment.AdjustmentStatus.PENDING))
                .collect(Collectors.toList());
    }

    @Override
    public Long saveAdjustment(BudgetAdjustment adjustment) {
        if (adjustment.getId() == null) {
            adjustment.setId(IdWorker.getId());
        }
        adjustments.put(adjustment.getId(), adjustment);
        return adjustment.getId();
    }

    @Override
    public void updateAdjustment(BudgetAdjustment adjustment) {
        if (adjustment.getId() != null) {
            adjustments.put(adjustment.getId(), adjustment);
        }
    }
}
