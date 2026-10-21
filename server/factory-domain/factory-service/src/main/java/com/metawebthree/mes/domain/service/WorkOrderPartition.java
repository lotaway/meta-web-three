package com.metawebthree.mes.domain.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;

import lombok.extern.slf4j.Slf4j;

/**
 * 工单拆分的领域逻辑。
 *
 * <p>职责：</p>
 * <ol>
 *   <li>校验拆分前置条件（父工单状态、规则启用状态、已有子工单等）；</li>
 *   <li>按 {@code BY_QUANTITY} / {@code MANUAL} 策略计算子工单数量（数量守恒）；</li>
 *   <li>派生子工单（保留父工单元数据、生成子工单号、绑定父子关系）；</li>
 *   <li>对 {@code BY_BOM} / {@code BY_PROCESS} 明确拒绝执行（能力未启用）。</li>
 * </ol>
 *
 * <p>数量守恒：子工单数量之和必须等于父工单数量，父工单数量本身保持不变，
 * 父工单在拆分后置为 {@link WorkOrder.WorkOrderStatus#CANCELLED}，避免
 * 父单生产与子单生产同时进行造成重复产出。</p>
 */
@Slf4j
public class WorkOrderPartition {

    private static final String SUPPORTED_SPLIT_TYPES = "BY_QUANTITY, MANUAL";

    private final WorkOrderRepository orders;
    private final WorkOrderSplitRuleRepository rules;

    public WorkOrderPartition(WorkOrderRepository orders, WorkOrderSplitRuleRepository rules) {
        this.orders = orders;
        this.rules = rules;
    }

    /**
     * 校验拆分前置条件。
     *
     * @param parent 父工单
     * @param rule   拆分规则
     */
    public static void validate(WorkOrder parent, WorkOrderSplitRule rule) {
        if (parent == null) {
            throw new IllegalArgumentException("Parent work order must not be null");
        }
        if (rule == null) {
            throw new IllegalArgumentException("Split rule must not be null");
        }
        if (!"DRAFT".equals(parent.getStatus() == null ? null : parent.getStatus().name())) {
            throw new IllegalStateException("Only DRAFT work orders can be split");
        }
        if (parent.getCompletedQuantity() != null && parent.getCompletedQuantity() > 0) {
            throw new IllegalStateException("Work order with completed production cannot be split");
        }
        if (rule.getEnabled() == null || !rule.getEnabled()) {
            throw new IllegalArgumentException("Persisted enabled split rule is required");
        }
    }

    /**
     * 执行拆分。
     *
     * @param parentWorkOrder 父工单（其 id 用于加锁读取权威状态）
     * @param splitRuleId     拆分规则 id
     * @param quantities      手动拆分时各子工单数量；批量拆分时必须为 null
     * @return 派生的子工单列表
     */
    public List<WorkOrder> execute(WorkOrder parentWorkOrder, Long splitRuleId, List<Integer> quantities) {
        WorkOrderSplitRule rule = rules.findById(splitRuleId)
                .orElseThrow(() -> new IllegalArgumentException("Split rule not found"));

        // 以仓储中的权威状态为准（父工单可能已被其他事务修改）
        WorkOrder parent = orders.findByIdForUpdate(parentWorkOrder.getId())
                .orElseThrow(() -> new IllegalArgumentException("Parent work order not found: " + parentWorkOrder.getId()));

        validate(parent, rule);
        rejectIfChildrenExist(parent.getId());

        List<WorkOrder> children;
        if ("BY_QUANTITY".equals(rule.getSplitType())) {
            if (quantities != null) {
                throw new IllegalArgumentException("Batch split does not accept manual quantities");
            }
            children = buildByQuantity(parent, rule);
        } else if ("MANUAL".equals(rule.getSplitType())) {
            children = buildManual(parent, rule, quantities);
        } else {
            throw new UnsupportedOperationException(
                    "Split strategy '" + rule.getSplitType() + "' is not implemented. "
                    + "Supported strategies: " + SUPPORTED_SPLIT_TYPES + ".");
        }

        // 落库：子工单逐个保存，父工单置为 CANCELLED 并保留原始数量
        for (WorkOrder child : children) {
            orders.save(child);
        }
        parent.setStatus(WorkOrder.WorkOrderStatus.CANCELLED);
        parent.setStatusCode(WorkOrder.WorkOrderStatus.CANCELLED.name());
        parent.setUpdatedAt(java.time.LocalDateTime.now());
        orders.save(parent);

        return children;
    }

    private void rejectIfChildrenExist(Long parentId) {
        List<WorkOrder> existing = orders.findByParentWorkOrderId(parentId);
        if (existing != null && !existing.isEmpty()) {
            throw new IllegalStateException(
                    "Work order " + parentId + " already has " + existing.size() + " child work orders");
        }
    }

    /**
     * 批量拆分：按规则中的批次数量切分，余数并入最后一个子工单，保证数量守恒。
     */
    private List<WorkOrder> buildByQuantity(WorkOrder parent, WorkOrderSplitRule rule) {
        Integer totalQuantity = parent.getQuantity();
        if (totalQuantity == null || totalQuantity <= 0) {
            throw new IllegalArgumentException("Parent work order must have positive quantity for batch split");
        }
        Integer batch = rule.getSplitQuantity();
        if (batch == null || batch <= 0) {
            throw new IllegalArgumentException("Positive splitQuantity is required");
        }

        int childCount = (int) Math.ceil((double) totalQuantity / batch);
        List<Integer> quantities = new ArrayList<>(childCount);
        int remaining = totalQuantity;
        for (int i = 0; i < childCount; i++) {
            int qty = Math.min(batch, remaining);
            quantities.add(qty);
            remaining -= qty;
        }
        return deriveChildren(parent, rule, quantities);
    }

    /**
     * 手动拆分：使用调用方给定的各子工单数量。
     */
    private List<WorkOrder> buildManual(WorkOrder parent, WorkOrderSplitRule rule, List<Integer> quantities) {
        if (quantities == null || quantities.isEmpty()) {
            throw new IllegalArgumentException("Manual split requires quantities for each child order");
        }
        if (quantities.size() < 2) {
            // 手动拆分成单个子工单等于没有拆分，属于无效请求
            throw new IllegalArgumentException(
                    "Manual split must produce at least two child work orders, got: " + quantities.size());
        }
        return deriveChildren(parent, rule, quantities);
    }

    /**
     * 派生子工单：校验数量并生成子工单对象。
     */
    private static List<WorkOrder> deriveChildren(WorkOrder parent, WorkOrderSplitRule rule, List<Integer> quantities) {
        Integer totalOriginal = parent.getQuantity();
        if (totalOriginal == null || totalOriginal <= 0) {
            throw new IllegalArgumentException("Parent work order must have positive quantity");
        }
        if (quantities == null || quantities.isEmpty()) {
            throw new IllegalArgumentException("Split must produce at least one child work order");
        }

        long totalRequested = 0;
        for (Integer q : quantities) {
            if (q == null || q <= 0) {
                throw new IllegalArgumentException("Child work order quantity must be positive, got: " + q);
            }
            totalRequested += q;
        }
        if (totalRequested != totalOriginal) {
            throw new IllegalArgumentException("Child work order quantities (" + totalRequested
                    + ") must sum up to parent quantity (" + totalOriginal + ")");
        }
        Integer maxChildOrders = rule.getMaxChildOrders();
        if (maxChildOrders != null && quantities.size() > maxChildOrders) {
            throw new IllegalArgumentException("Child work order count (" + quantities.size()
                    + ") exceeds maxChildOrders (" + maxChildOrders + ")");
        }

        List<WorkOrder> children = new ArrayList<>(quantities.size());
        int sequence = 1;
        for (Integer qty : quantities) {
            WorkOrder child = new WorkOrder();
            child.createWithType(childWorkOrderNo(parent.getWorkOrderNo(), sequence),
                    parent.getProductCode(), parent.getProductName(), qty,
                    parent.getWorkshopId(), parent.getProcessRouteId(), parent.getTypeCode());
            child.setParentWorkOrderId(parent.getId());
            child.setSplitRuleId(rule.getId());
            child.setSplitSequence(sequence);
            child.setSplitType(rule.getSplitType());
            child.setPriority(parent.getPriority());
            child.setCodeRuleId(parent.getCodeRuleId());
            child.setPlannedStartTime(parent.getPlannedStartTime());
            child.setPlannedEndTime(parent.getPlannedEndTime());
            children.add(child);
            sequence++;
        }
        return children;
    }

    private static String childWorkOrderNo(String parentNo, int sequence) {
        return parentNo + "-S" + String.format("%02d", sequence);
    }

    /**
     * 校验拆分条件。
     *
     * <p>支持的条件字段为 {@code quantity} 与 {@code completedQuantity}，
     * 操作符支持 {@code EQ} / {@code GTE} / {@code GT} / {@code LTE} / {@code LT}。
     * 条件之间使用 AND 逻辑；出现 OR 逻辑时明确拒绝（尚未支持）。</p>
     */
    static boolean conditionsMatch(WorkOrder order, WorkOrderSplitRule rule) {
        List<WorkOrderSplitRule.SplitCondition> conditions = rule.getConditions();
        if (conditions == null || conditions.isEmpty()) {
            return true;
        }
        for (WorkOrderSplitRule.SplitCondition c : conditions) {
            if (c.getLogic() != null && !"AND".equalsIgnoreCase(c.getLogic())) {
                throw new IllegalArgumentException("Unsupported split condition logic: " + c.getLogic());
            }
            String field = c.getField();
            if (field == null) {
                throw new IllegalArgumentException("Split condition field must not be null");
            }
            Long actual = resolveField(order, field);
            Object expected = c.getValue();
            if (expected == null) {
                // 值为空的条件视为不满足
                return false;
            }
            long expectedValue = toLong(expected, field);
            if (!compare(actual, expectedValue, c.getOperator())) {
                return false;
            }
        }
        return true;
    }

    private static Long resolveField(WorkOrder order, String field) {
        switch (field) {
            case "quantity":
                return order.getQuantity() == null ? null : order.getQuantity().longValue();
            case "completedQuantity":
            case "completed_quantity":
                return order.getCompletedQuantity() == null ? null : order.getCompletedQuantity().longValue();
            default:
                throw new IllegalArgumentException("Unsupported split condition field: " + field);
        }
    }

    private static long toLong(Object value, String field) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Split condition value for field '" + field + "' is not numeric: " + value, e);
        }
    }

    private static boolean compare(Long actual, long expected, String operator) {
        if (actual == null) {
            return false;
        }
        String op = operator == null ? "EQ" : operator.toUpperCase();
        switch (op) {
            case "EQ":
                return actual == expected;
            case "GTE":
                return actual >= expected;
            case "GT":
                return actual > expected;
            case "LTE":
                return actual <= expected;
            case "LT":
                return actual < expected;
            default:
                throw new IllegalArgumentException("Unsupported split condition operator: " + operator);
        }
    }

    /**
     * 预览拆分结果：返回将要生成的子工单，但不落库。
     *
     * <p>仅支持 {@code BY_QUANTITY} 与 {@code MANUAL}；其他策略明确拒绝。</p>
     *
     * @param parent     父工单
     * @param rule       拆分规则
     * @param quantities 手动拆分时各子工单数量（{@code BY_QUANTITY} 时忽略）
     * @return 子工单预览列表
     */
    public static List<WorkOrder> preview(WorkOrder parent, WorkOrderSplitRule rule, List<Integer> quantities) {
        if (parent == null) {
            throw new IllegalArgumentException("Parent work order must not be null");
        }
        if (rule == null) {
            throw new IllegalArgumentException("Split rule must not be null");
        }
        if (!"BY_QUANTITY".equals(rule.getSplitType()) && !"MANUAL".equals(rule.getSplitType())) {
            throw new UnsupportedOperationException(
                    "Preview only supported for BY_QUANTITY and MANUAL split types. "
                    + "BY_BOM and BY_PROCESS are not yet available.");
        }
        validate(parent, rule);

        if ("BY_QUANTITY".equals(rule.getSplitType())) {
            Integer totalQty = parent.getQuantity();
            Integer batch = rule.getSplitQuantity();
            if (totalQty == null || totalQty <= 0) {
                throw new IllegalArgumentException("Parent work order must have positive quantity for batch split");
            }
            if (batch == null || batch <= 0) {
                throw new IllegalArgumentException("Split rule must have positive splitQuantity");
            }
            int childCount = (int) Math.ceil((double) totalQty / batch);
            List<Integer> derived = new ArrayList<>(childCount);
            int remaining = totalQty;
            for (int i = 0; i < childCount; i++) {
                int qty = Math.min(batch, remaining);
                derived.add(qty);
                remaining -= qty;
            }
            return deriveChildren(parent, rule, derived);
        }

        if (quantities == null || quantities.isEmpty()) {
            throw new IllegalArgumentException("Manual split requires at least one quantity");
        }
        if (quantities.size() < 2) {
            throw new IllegalArgumentException(
                    "Manual split must produce at least two child work orders, got: " + quantities.size());
        }
        return deriveChildren(parent, rule, quantities);
    }
}
