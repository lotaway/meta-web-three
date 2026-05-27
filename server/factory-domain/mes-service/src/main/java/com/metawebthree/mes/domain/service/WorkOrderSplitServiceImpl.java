package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule.SplitCondition;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class WorkOrderSplitServiceImpl implements WorkOrderSplitService {

    private final WorkOrderSplitRuleRepository splitRuleRepository;
    private final WorkOrderRepository workOrderRepository;
    private final CodeRuleService codeRuleService;

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
        if (parentWorkOrder == null) {
            throw new IllegalArgumentException("Parent work order cannot be null");
        }
        if (parentWorkOrder.getId() == null) {
            throw new IllegalArgumentException("Parent work order must be saved first");
        }
        if (!parentWorkOrder.getStatus().equals(WorkOrder.WorkOrderStatus.DRAFT)) {
            throw new IllegalStateException("Only DRAFT work orders can be split");
        }

        Optional<WorkOrderSplitRule> ruleOpt = splitRuleRepository.findById(splitRuleId);
        if (ruleOpt.isEmpty()) {
            throw new IllegalArgumentException("Split rule not found: " + splitRuleId);
        }

        WorkOrderSplitRule splitRule = ruleOpt.get();
        if (!splitRule.getEnabled()) {
            throw new IllegalStateException("Split rule is disabled: " + splitRule.getRuleName());
        }

        if (!validateSplitConditions(parentWorkOrder, splitRule)) {
            throw new IllegalStateException("Split conditions not met");
        }

        List<WorkOrder> childOrders;
        String splitType = splitRule.getSplitType();

        if (WorkOrder.SplitType.BY_BOM.name().equals(splitType)) {
            childOrders = splitByBom(parentWorkOrder, splitRule);
        } else if (WorkOrder.SplitType.BY_PROCESS.name().equals(splitType)) {
            childOrders = splitByProcess(parentWorkOrder, splitRule);
        } else if (WorkOrder.SplitType.MANUAL.name().equals(splitType)) {
            throw new IllegalArgumentException("Manual split requires quantities parameter");
        } else {
            throw new IllegalArgumentException("Unknown split type: " + splitType);
        }

        for (WorkOrder child : childOrders) {
            workOrderRepository.save(child);
        }

        parentWorkOrder.setStatus(WorkOrder.WorkOrderStatus.CANCELLED);
        workOrderRepository.save(parentWorkOrder);

        return childOrders;
    }

    @Override
    public List<WorkOrder> splitByBom(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule) {
        List<WorkOrder> childOrders = new ArrayList<>();
        String bomId = splitRule.getBomId();

        if (bomId == null) {
            throw new IllegalStateException("BOM ID is required for BY_BOM split type");
        }

        Integer splitQty = splitRule.getSplitQuantity();
        Integer maxChildren = splitRule.getMaxChildOrders();
        int totalQty = parentWorkOrder.getQuantity();

        if (splitQty == null || splitQty <= 0) {
            splitQty = totalQty;
        }
        if (maxChildren == null || maxChildren <= 0) {
            maxChildren = 100;
        }

        int childCount = (int) Math.ceil((double) totalQty / splitQty);
        childCount = Math.min(childCount, maxChildren);

        String baseNo = parentWorkOrder.getWorkOrderNo();

        for (int i = 1; i <= childCount; i++) {
            int qty = Math.min(splitQty, totalQty - (i - 1) * splitQty);
            WorkOrder child = createChildWorkOrder(parentWorkOrder, splitRule, i, qty, baseNo);
            childOrders.add(child);
            parentWorkOrder.setQuantity(parentWorkOrder.getQuantity() - qty);
        }

        return childOrders;
    }

    @Override
    public List<WorkOrder> splitByProcess(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule) {
        List<WorkOrder> childOrders = new ArrayList<>();
        String processRouteId = splitRule.getProcessRouteId();

        if (processRouteId == null) {
            throw new IllegalStateException("Process route ID is required for BY_PROCESS split type");
        }

        int processCount = 3;
        int totalQty = parentWorkOrder.getQuantity();
        int qtyPerProcess = totalQty / processCount;
        int remainder = totalQty % processCount;

        String baseNo = parentWorkOrder.getWorkOrderNo();

        for (int i = 1; i <= processCount; i++) {
            int qty = qtyPerProcess + (i == 1 ? remainder : 0);
            WorkOrder child = createChildWorkOrder(parentWorkOrder, splitRule, i, qty, baseNo);
            child.setProcessRouteId(processRouteId);
            childOrders.add(child);
        }

        return childOrders;
    }

    @Override
    public List<WorkOrder> splitManually(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule,
                                          List<Integer> quantities) {
        if (quantities == null || quantities.isEmpty()) {
            throw new IllegalArgumentException("Quantities list cannot be empty for manual split");
        }

        Integer maxChildren = splitRule.getMaxChildOrders();
        if (maxChildren != null && quantities.size() > maxChildren) {
            throw new IllegalStateException("Exceeds maximum child work orders: " + maxChildren);
        }

        int totalRequested = quantities.stream().mapToInt(Integer::intValue).sum();
        if (totalRequested != parentWorkOrder.getQuantity()) {
            throw new IllegalArgumentException(
                "Total quantities " + totalRequested + " does not match parent quantity " 
                + parentWorkOrder.getQuantity());
        }

        List<WorkOrder> childOrders = new ArrayList<>();
        String baseNo = parentWorkOrder.getWorkOrderNo();

        for (int i = 0; i < quantities.size(); i++) {
            WorkOrder child = createChildWorkOrder(parentWorkOrder, splitRule, i + 1, 
                                                    quantities.get(i), baseNo);
            childOrders.add(child);
        }

        return childOrders;
    }

    @Override
    public List<WorkOrder> getChildWorkOrders(Long parentWorkOrderId) {
        return workOrderRepository.findAll().stream()
            .filter(wo -> parentWorkOrderId.equals(wo.getParentWorkOrderId()))
            .collect(Collectors.toList());
    }

    @Override
    public WorkOrder mergeChildWorkOrders(Long parentWorkOrderId) {
        List<WorkOrder> children = getChildWorkOrders(parentWorkOrderId);
        
        if (children.isEmpty()) {
            throw new IllegalStateException("No child work orders to merge");
        }

        Optional<WorkOrder> parentOpt = workOrderRepository.findById(parentWorkOrderId);
        if (parentOpt.isEmpty()) {
            throw new IllegalArgumentException("Parent work order not found: " + parentWorkOrderId);
        }

        WorkOrder parent = parentOpt.get();
        
        int totalCompleted = children.stream()
            .mapToInt(WorkOrder::getCompletedQuantity)
            .sum();
        parent.setCompletedQuantity(totalCompleted);

        boolean allCompleted = children.stream()
            .allMatch(c -> c.getStatus().equals(WorkOrder.WorkOrderStatus.COMPLETED));
        
        if (allCompleted) {
            parent.setStatus(WorkOrder.WorkOrderStatus.COMPLETED);
            parent.setActualEndTime(LocalDateTime.now());
        } else {
            parent.setStatus(WorkOrder.WorkOrderStatus.IN_PROGRESS);
        }

        workOrderRepository.save(parent);

        for (WorkOrder child : children) {
            child.setStatus(WorkOrder.WorkOrderStatus.CANCELLED);
            workOrderRepository.save(child);
        }

        return parent;
    }

    @Override
    public boolean validateSplitConditions(WorkOrder workOrder, WorkOrderSplitRule splitRule) {
        List<SplitCondition> conditions = splitRule.getConditions();
        
        if (conditions == null || conditions.isEmpty()) {
            return true;
        }

        for (SplitCondition condition : conditions) {
            if (!evaluateCondition(workOrder, condition)) {
                return false;
            }
        }
        return true;
    }

    private boolean evaluateCondition(WorkOrder workOrder, SplitCondition condition) {
        Object fieldValue = getFieldValue(workOrder, condition.getField());
        Object targetValue = condition.getValue();
        String operator = condition.getOperator();

        switch (operator) {
            case "EQ":
                return fieldValue != null && fieldValue.equals(targetValue);
            case "GT":
                return compareValues(fieldValue, targetValue) > 0;
            case "LT":
                return compareValues(fieldValue, targetValue) < 0;
            case "GTE":
                return compareValues(fieldValue, targetValue) >= 0;
            case "LTE":
                return compareValues(fieldValue, targetValue) <= 0;
            case "IN":
                if (targetValue instanceof List) {
                    return ((List<?>) targetValue).contains(fieldValue);
                }
                return false;
            default:
                return false;
        }
    }

    private Object getFieldValue(WorkOrder workOrder, String field) {
        switch (field) {
            case "quantity":
                return workOrder.getQuantity();
            case "productCode":
                return workOrder.getProductCode();
            case "workshopId":
                return workOrder.getWorkshopId();
            case "typeCode":
                return workOrder.getTypeCode();
            case "priority":
                return workOrder.getPriority() != null ? workOrder.getPriority().name() : null;
            default:
                return null;
        }
    }

    @SuppressWarnings("unchecked")
    private int compareValues(Object v1, Object v2) {
        if (v1 == null || v2 == null) return 0;
        
        if (v1 instanceof Comparable && v2 instanceof Comparable) {
            return ((Comparable<Object>) v1).compareTo(v2);
        }
        return 0;
    }

    private WorkOrder createChildWorkOrder(WorkOrder parent, WorkOrderSplitRule rule, 
                                            int sequence, int quantity, String baseNo) {
        WorkOrder child = new WorkOrder();
        
        String childNo = baseNo + "-S" + String.format("%02d", sequence);
        if (codeRuleService != null) {
            childNo = codeRuleService.generateCode("WORK_ORDER_SPLIT");
        }
        
        child.create(childNo, parent.getProductCode(), parent.getProductName(),
                     quantity, parent.getWorkshopId(), parent.getProcessRouteId());
        
        child.setParentWorkOrderId(parent.getId());
        child.setSplitRuleId(rule.getId());
        child.setSplitSequence(sequence);
        child.setSplitType(rule.getSplitType());
        child.setPriority(parent.getPriority());
        child.setPlannedStartTime(parent.getPlannedStartTime());
        child.setPlannedEndTime(parent.getPlannedEndTime());
        
        return child;
    }

    public interface CodeRuleService {
        String generateCode(String ruleCode);
    }
}