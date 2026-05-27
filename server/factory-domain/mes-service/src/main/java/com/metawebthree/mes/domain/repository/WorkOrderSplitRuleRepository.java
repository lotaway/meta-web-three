package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import java.util.List;
import java.util.Optional;

public interface WorkOrderSplitRuleRepository {
    WorkOrderSplitRule save(WorkOrderSplitRule rule);
    Optional<WorkOrderSplitRule> findById(Long id);
    Optional<WorkOrderSplitRule> findByRuleCode(String ruleCode);
    List<WorkOrderSplitRule> findAll();
    List<WorkOrderSplitRule> findByEnabled(Boolean enabled);
    List<WorkOrderSplitRule> findBySplitType(String splitType);
    void deleteById(Long id);
}