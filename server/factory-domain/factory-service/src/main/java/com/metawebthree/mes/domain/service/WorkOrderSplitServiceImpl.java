package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@org.springframework.transaction.annotation.Transactional
public class WorkOrderSplitServiceImpl implements WorkOrderSplitService {

    private final WorkOrderSplitRuleRepository splitRuleRepository;
    private final WorkOrderRepository workOrderRepository;
    private final CodeRuleService codeRuleService;

    @org.springframework.beans.factory.annotation.Autowired
    public WorkOrderSplitServiceImpl(WorkOrderSplitRuleRepository rules, WorkOrderRepository orders) {
        this(rules, orders, null);
    }

    public WorkOrderSplitServiceImpl(
            WorkOrderSplitRuleRepository splitRuleRepository,
            WorkOrderRepository workOrderRepository,
            CodeRuleService codeRuleService) {
        this.splitRuleRepository = splitRuleRepository;
        this.workOrderRepository = workOrderRepository;
        this.codeRuleService = codeRuleService;
    }

    @Override
    public List<WorkOrder> splitWorkOrder(WorkOrder parentWorkOrder, Long splitRuleId) {
        return new WorkOrderPartition(workOrderRepository, splitRuleRepository)
                .execute(parentWorkOrder, splitRuleId, null);
    }

    @Override
    public List<WorkOrder> splitByBom(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule) {
        throw new SplitNotSupportedException(
                "BY_BOM split strategy is not yet implemented. "
                + "Real BOM expansion requires material structure resolution. "
                + "See TODO.md for planned implementation.");
    }

    @Override
    public List<WorkOrder> splitByProcess(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule) {
        throw new SplitNotSupportedException(
                "BY_PROCESS split strategy is not yet implemented. "
                + "Requires routing task decomposition and operation sequencing. "
                + "See TODO.md for planned implementation.");
    }

    @Override
    public List<WorkOrder> splitManually(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule,
                                           List<Integer> quantities) {
        if (splitRule == null || !"MANUAL".equals(splitRule.getSplitType()))
            throw new IllegalArgumentException("Manual split requires a MANUAL rule");
        return new WorkOrderPartition(workOrderRepository, splitRuleRepository)
                .execute(parentWorkOrder, splitRule.getId(), quantities);
    }

    @Override
    public List<WorkOrder> getChildWorkOrders(Long parentWorkOrderId) {
        return workOrderRepository.findByParentWorkOrderId(parentWorkOrderId);
    }

    @Override
    public WorkOrder mergeChildWorkOrders(Long parentWorkOrderId) {
        throw new SplitNotSupportedException(
                "Merge/aggregation semantics are not yet implemented. "
                + "Requires reconciled completion quantities and cost reallocation. "
                + "See TODO.md for planned implementation.");
    }

    @Override
    public boolean validateSplitConditions(WorkOrder workOrder, WorkOrderSplitRule splitRule) {
        return WorkOrderPartition.conditionsMatch(workOrder, splitRule);
    }

    public interface CodeRuleService {
        String generateCode(String ruleCode);
    }
}