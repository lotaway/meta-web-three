package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;

import java.util.List;

@org.springframework.stereotype.Service
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
        throw new UnsupportedOperationException("BY_BOM requires real BOM expansion; see TODO.md");
    }

    @Override
    public List<WorkOrder> splitByProcess(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule) {
        throw new UnsupportedOperationException("BY_PROCESS requires routing task decomposition; see TODO.md");
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
        throw new UnsupportedOperationException("Merge/aggregation semantics are not implemented; see TODO.md");
    }

    @Override
    public boolean validateSplitConditions(WorkOrder workOrder, WorkOrderSplitRule splitRule) {
        return WorkOrderPartition.conditionsMatch(workOrder, splitRule);
    }

    public interface CodeRuleService {
        String generateCode(String ruleCode);
    }
}