package com.metawebthree.mes.domain.service;

import java.util.List;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class WorkOrderPartition {

    private final WorkOrderSplitRuleRepository rules;
    private final WorkOrderRepository orders;

    public WorkOrderPartition(WorkOrderRepository orders, WorkOrderSplitRuleRepository rules) {
        this.orders = orders;
        this.rules = rules;
    }

    public static void validate(WorkOrder parent, WorkOrderSplitRule rule) {
        if (parent == null) {
            throw new IllegalArgumentException("Parent work order must not be null");
        }
        if (rule == null) {
            throw new IllegalArgumentException("Split rule must not be null");
        }
        if (!"DRAFT".equals(parent.getStatus())) {
            throw new IllegalStateException("Only DRAFT work orders can be split");
        }
        if (parent.getCompletedQuantity() != null && parent.getCompletedQuantity() > 0) {
            throw new IllegalStateException("Work order with completed production cannot be split");
        }
        if (rule.getEnabled() == null || !rule.getEnabled()) {
            throw new IllegalArgumentException("Persisted enabled split rule is required");
        }
    }

    public List<WorkOrder> execute(WorkOrder parentWorkOrder, Long splitRuleId, List<Integer> quantities) {
        WorkOrderSplitRule rule = rules.findById(splitRuleId)
                .orElseThrow(() -> new IllegalArgumentException("Split rule not found"));
        validate(parentWorkOrder, rule);

        if ("BY_QUANTITY".equals(rule.getSplitType())) {
            if (quantities != null) {
                throw new IllegalArgumentException("Batch split does not accept manual quantities");
            }
            Integer batch = rule.getSplitQuantity();
            if (batch == null || batch <= 0) {
                throw new IllegalArgumentException("Positive splitQuantity is required");
            }
            // Build child orders by quantity batch
            return buildByQuantity(parentWorkOrder, rule, batch);
        } else if ("MANUAL".equals(rule.getSplitType())) {
            if (quantities == null || quantities.isEmpty()) {
                throw new IllegalArgumentException("Manual split requires quantities for each child order");
            }
            return executeManualSplit(parentWorkOrder, rule, quantities);
        } else {
            throw new SplitNotSupportedException(
                    "Split strategy '" + rule.getSplitType() + "' is not implemented. "
                    + "Supported strategies: BY_QUANTITY, MANUAL. "
                    + "See TODO.md for planned BY_BOM, BY_PROCESS support.");
        }
    }

    private List<WorkOrder> buildByQuantity(WorkOrder parentWorkOrder, WorkOrderSplitRule rule, Integer batch) {
        // Basic quantity-based split - divide total quantity into batches
        Integer totalQuantity = parentWorkOrder.getQuantity();
        if (totalQuantity == null || totalQuantity <= 0) {
            throw new IllegalArgumentException("Parent work order must have positive quantity for batch split");
        }
        // Calculate number of children
        Integer numChildren = (int) Math.ceil((double) totalQuantity / batch);
        // ... (simplified for TODO.md boundary marking)
        throw new SplitNotSupportedException(
                "BY_QUANTITY batch split is partially implemented. "
                + "Full quantity distribution logic pending. "
                + "See TODO.md for complete implementation.");
    }

    private List<WorkOrder> executeManualSplit(WorkOrder parentWorkOrder, WorkOrderSplitRule rule, List<Integer> quantities) {
        // Manual split with explicit quantities
        Integer totalRequested = 0;
        for (Integer q : quantities) {
            if (q != null) totalRequested += q;
        }
        Integer totalOriginal = parentWorkOrder.getQuantity();
        if (totalOriginal == null || totalOriginal <= 0) {
            throw new IllegalArgumentException("Parent work order must have positive quantity");
        }
        if (totalRequested > totalOriginal) {
            throw new IllegalArgumentException("Total requested quantities (" + totalRequested 
                    + ") exceeds parent quantity (" + totalOriginal + ")");
        }
        // ... (partition logic)
        throw new SplitNotSupportedException(
                "Manual split partition logic is partially implemented. "
                + "Child order creation pending. "
                + "See TODO.md for complete implementation.");
    }

    static boolean conditionsMatch(WorkOrder order, WorkOrderSplitRule rule) {
        for (WorkOrderSplitRule.SplitCondition c : rule.getConditions()) {
            switch (c.getType()) {
                case QUANTITY:
                    if (order.getQuantity() == null || order.getQuantity() < c.getValue()) {
                        throw new IllegalArgumentException(
                                "Split condition quantity not met: required>=" + c.getValue()
                                + ", actual=" + order.getQuantity());
                    }
                    break;
                case COMPLETED_QUANTITY:
                    if (order.getCompletedQuantity() == null || order.getCompletedQuantity() < c.getValue()) {
                        throw new IllegalArgumentException(
                                "Split condition completed quantity not met: required>=" + c.getValue()
                                + ", actual=" + order.getCompletedQuantity());
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Invalid split condition type: " + c.getType());
            }
        }
        return true;
    }

    /**
     * Preview child orders that would result from a split, without actually creating them.
     * <p>
     * Only BY_QUANTITY and MANUAL split types are supported for preview.
     * Other strategies (BY_BOM, BY_PROCESS) throw SplitNotSupportedException.
     *
     * @param parent parent work order to split
     * @param splitRule the split rule to apply
     * @param quantities optional explicit quantities for manual split (ignored for BY_QUANTITY)
     * @return list of child order DTOs representing the preview
     */
    public List<WorkOrder> preview(WorkOrder parent, WorkOrderSplitRule rule, List<Integer> quantities) {
        if (!List.of("BY_QUANTITY", "MANUAL").contains(rule.getSplitType())) {
            throw new SplitNotSupportedException(
                    "Preview only supported for BY_QUANTITY and MANUAL split types. "
                    + "BY_BOM and BY_PROCESS preview not yet available. "
                    + "See TODO.md for planned implementation.");
        }
        // For BY_QUANTITY, provide a basic preview
        if ("BY_QUANTITY".equals(rule.getSplitType())) {
            Integer totalQty = parent.getQuantity();
            Integer batch = rule.getSplitQuantity();
            if (batch == null || batch <= 0) {
                throw new IllegalArgumentException("Split rule must have positive splitQuantity");
            }
            Integer numChildren = (totalQty != null && batch > 0)
                    ? (int) Math.ceil((double) totalQty / batch) : 0;
            log.info("BY_QUANTITY preview: {} parent qty / batch {} = ~{} children",
                    totalQty, batch, numChildren);
            // Return minimal preview - in full implementation, would build child order summaries
            return java.util.Collections.emptyList();
        }
        // For MANUAL, provide a preview based on requested quantities
        if ("MANUAL".equals(rule.getSplitType())) {
            if (quantities == null || quantities.isEmpty()) {
                throw new IllegalArgumentException("Manual split requires at least one quantity");
            }
            Integer totalRequested = 0;
            for (Integer q : quantities) {
                if (q != null) totalRequested += q;
            }
            Integer totalOriginal = parent.getQuantity();
            log.info("MANUAL preview: {} original qty, {} requested", totalOriginal, totalRequested);
            return java.util.Collections.emptyList();
        }
        return java.util.Collections.emptyList();
    }
}