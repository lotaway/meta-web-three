package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Quantity partitions only: BOM expansion and routing are different operations. */
final class WorkOrderPartition {
    private final WorkOrderRepository orders;
    private final WorkOrderSplitRuleRepository rules;

    WorkOrderPartition(WorkOrderRepository orders, WorkOrderSplitRuleRepository rules) {
        this.orders = orders;
        this.rules = rules;
    }

    List<WorkOrder> execute(WorkOrder requestedParent, Long ruleId, List<Integer> quantities) {
        if (requestedParent == null || requestedParent.getId() == null || ruleId == null)
            throw new IllegalArgumentException("Persisted parent and rule ID are required");
        WorkOrder parent = orders.findByIdForUpdate(requestedParent.getId())
                .orElseThrow(() -> new IllegalArgumentException("Parent work order not found"));
        WorkOrderSplitRule rule = rules.findById(ruleId)
                .orElseThrow(() -> new IllegalArgumentException("Split rule not found"));
        validate(parent, rule);
        if (!orders.findByParentWorkOrderId(parent.getId()).isEmpty())
            throw new IllegalStateException("Work order has already been split");
        List<Integer> parts;
        if ("BY_QUANTITY".equals(rule.getSplitType())) {
            if (quantities != null) throw new IllegalArgumentException("Batch split does not accept manual quantities");
            Integer batch = rule.getSplitQuantity();
            if (batch == null || batch <= 0) throw new IllegalArgumentException("Positive splitQuantity is required");
            long count = ((long) parent.getQuantity() + batch - 1) / batch;
            validateCount(count, rule);
            parts = new ArrayList<>();
            for (int remaining = parent.getQuantity(); remaining > 0;) {
                int qty = Math.min(batch, remaining);
                parts.add(qty);
                remaining -= qty;
            }
        } else if ("MANUAL".equals(rule.getSplitType())) {
            parts = quantities;
        } else {
            throw new UnsupportedOperationException("Split strategy " + rule.getSplitType() + " is not implemented; see TODO.md");
        }
        List<WorkOrder> children = preview(parent, rule, parts);
        for (WorkOrder child : children) orders.save(child);
        parent.setSplitRuleId(rule.getId());
        parent.setSplitType(rule.getSplitType());
        parent.setStatus(WorkOrder.WorkOrderStatus.CANCELLED);
        orders.save(parent);
        return children;
    }

    static void validate(WorkOrder parent, WorkOrderSplitRule rule) {
        if (parent == null || parent.getId() == null || parent.getQuantity() == null || parent.getQuantity() <= 0)
            throw new IllegalArgumentException("Persisted parent with positive quantity is required");
        if (parent.getStatus() != WorkOrder.WorkOrderStatus.DRAFT)
            throw new IllegalStateException("Only DRAFT work orders can be split");
        if (parent.getCompletedQuantity() != null && parent.getCompletedQuantity() != 0)
            throw new IllegalStateException("Work order with completed production cannot be split");
        if (rule == null || rule.getId() == null || !Boolean.TRUE.equals(rule.getEnabled()))
            throw new IllegalArgumentException("Persisted enabled split rule is required");
        if (!conditionsMatch(parent, rule)) throw new IllegalStateException("Split conditions not met");
    }

    static List<WorkOrder> preview(WorkOrder parent, WorkOrderSplitRule rule, List<Integer> quantities) {
        validate(parent, rule);
        if (!List.of("BY_QUANTITY", "MANUAL").contains(rule.getSplitType()))
            throw new UnsupportedOperationException("Only BY_QUANTITY and MANUAL are supported");
        if (quantities == null) throw new IllegalArgumentException("Quantities are required");
        validateCount(quantities.size(), rule);
        long total = 0;
        for (Integer qty : quantities) {
            if (qty == null || qty <= 0) throw new IllegalArgumentException("Each quantity must be positive");
            total += qty;
        }
        if (total != parent.getQuantity()) throw new IllegalArgumentException("Child quantities must equal parent quantity");
        if (parent.getWorkOrderNo() == null || parent.getWorkOrderNo().isBlank())
            throw new IllegalArgumentException("Parent work order number is required");
        List<WorkOrder> children = new ArrayList<>();
        for (int i = 0; i < quantities.size(); i++) {
            WorkOrder child = new WorkOrder();
            child.createWithType(parent.getWorkOrderNo() + "-S" + String.format("%02d", i + 1),
                    parent.getProductCode(), parent.getProductName(), quantities.get(i),
                    parent.getWorkshopId(), parent.getProcessRouteId(), parent.getTypeCode());
            child.setParentWorkOrderId(parent.getId());
            child.setSplitRuleId(rule.getId());
            child.setSplitSequence(i + 1);
            child.setSplitType(rule.getSplitType());
            child.setPriority(parent.getPriority());
            child.setPlannedStartTime(parent.getPlannedStartTime());
            child.setPlannedEndTime(parent.getPlannedEndTime());
            children.add(child);
        }
        return children;
    }

    private static void validateCount(long count, WorkOrderSplitRule rule) {
        int limit = rule.getMaxChildOrders() == null ? 100 : rule.getMaxChildOrders();
        if (limit < 2 || limit > 1000 || count < 2 || count > limit)
            throw new IllegalArgumentException("Child count must be between 2 and configured maximum (at most 1000)");
    }

    static boolean conditionsMatch(WorkOrder order, WorkOrderSplitRule rule) {
        if (rule.getConditions() == null) return true;
        for (WorkOrderSplitRule.SplitCondition c : rule.getConditions()) {
            if (c == null || c.getField() == null || c.getOperator() == null)
                throw new IllegalArgumentException("Invalid split condition");
            if (c.getLogic() != null && !"AND".equals(c.getLogic()))
                throw new IllegalArgumentException("Only AND conditions are supported");
            Object actual = switch (c.getField()) {
                case "quantity" -> order.getQuantity();
                case "productCode" -> order.getProductCode();
                case "workshopId" -> order.getWorkshopId();
                case "typeCode" -> order.getTypeCode();
                case "priority" -> order.getPriority() == null ? null : order.getPriority().name();
                default -> throw new IllegalArgumentException("Unknown condition field: " + c.getField());
            };
            Object target = c.getValue();
            boolean matches;
            switch (c.getOperator()) {
                case "EQ" -> matches = equal(actual, target);
                case "IN" -> {
                    if (!(target instanceof List<?> list)) throw new IllegalArgumentException("IN requires a list");
                    matches = list.stream().anyMatch(v -> equal(actual, v));
                }
                case "GT", "LT", "GTE", "LTE" -> {
                    if (actual == null || target == null) return false;
                    if (!(actual instanceof Number) || !(target instanceof Number))
                        throw new IllegalArgumentException("Ordered conditions require numeric values");
                    int cmp = new BigDecimal(actual.toString()).compareTo(new BigDecimal(target.toString()));
                    matches = switch (c.getOperator()) { case "GT" -> cmp > 0; case "LT" -> cmp < 0; case "GTE" -> cmp >= 0; default -> cmp <= 0; };
                }
                default -> throw new IllegalArgumentException("Unknown condition operator: " + c.getOperator());
            }
            if (!matches) return false;
        }
        return true;
    }

    private static boolean equal(Object a, Object b) {
        if (a == null || b == null) return false;
        if (a instanceof Number && b instanceof Number)
            return new BigDecimal(a.toString()).compareTo(new BigDecimal(b.toString())) == 0;
        return Objects.equals(a, b);
    }
}
