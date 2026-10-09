package com.metawebthree.finance.infrastructure.persistence.repository.cash;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cash.CashPlan;
import com.metawebthree.finance.domain.entity.cash.CashPlanLine;
import com.metawebthree.finance.domain.repository.cash.CashPlanRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 资金计划仓储（本地开发用内存实现）。
 */
@Repository
public class CashPlanRepositoryImpl implements CashPlanRepository {

    private final Map<Long, CashPlan> plans = new ConcurrentHashMap<>();
    private final Map<Long, CashPlanLine> lines = new ConcurrentHashMap<>();

    @Override
    public Long save(CashPlan cashPlan) {
        if (cashPlan.getId() == null) {
            cashPlan.setId(IdWorker.getId());
        }
        plans.put(cashPlan.getId(), cashPlan);
        return cashPlan.getId();
    }

    @Override
    public void update(CashPlan cashPlan) {
        if (cashPlan.getId() != null) {
            plans.put(cashPlan.getId(), cashPlan);
        }
    }

    @Override
    public Optional<CashPlan> findById(Long id) {
        return Optional.ofNullable(plans.get(id));
    }

    @Override
    public Optional<CashPlan> findByPlanCode(String planCode) {
        return plans.values().stream()
                .filter(p -> Objects.equals(p.getPlanCode(), planCode))
                .findFirst();
    }

    @Override
    public List<CashPlan> findAll() {
        return plans.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CashPlan> findByStatus(CashPlan.CashPlanStatus status) {
        return plans.values().stream()
                .filter(p -> Objects.equals(p.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashPlan> findByDepartmentId(Long departmentId) {
        return plans.values().stream()
                .filter(p -> Objects.equals(p.getDepartmentId(), departmentId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashPlan> findByPeriod(CashPlan.CashPlanPeriod period) {
        return plans.values().stream()
                .filter(p -> Objects.equals(p.getPeriod(), period))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        plans.remove(id);
    }

    @Override
    public Long saveLine(CashPlanLine line) {
        if (line.getId() == null) {
            line.setId(IdWorker.getId());
        }
        lines.put(line.getId(), line);
        return line.getId();
    }

    @Override
    public List<CashPlanLine> findLinesByCashPlanId(Long cashPlanId) {
        return lines.values().stream()
                .filter(l -> Objects.equals(l.getCashPlanId(), cashPlanId))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteLine(Long lineId) {
        lines.remove(lineId);
    }
}
