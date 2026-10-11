package com.metawebthree.mes.domain.entity.scheduling;

import java.time.LocalDateTime;
import java.util.List;

public class ScheduleResult {

    public enum ScheduleStatus {
        SUCCESS, PARTIAL, FAILED
    }

    public enum ScheduleDirection {
        FORWARD, BACKWARD
    }

    private ScheduleStatus status;
    private ScheduleDirection direction;
    private List<ScheduleOrder> scheduledOrders;
    private List<ScheduleConflict> conflicts;
    private LocalDateTime computedAt;
    private long computationTimeMs;
    private int totalOrders;
    private int scheduledCount;
    private int failedCount;

    public static class ScheduleConflict {
        private String orderNo;
        private String productCode;
        private String resourceCode;
        private String constraintType;
        private String description;

        public ScheduleConflict(String orderNo, String productCode, String resourceCode,
                                 String constraintType, String description) {
            this.orderNo = orderNo;
            this.productCode = productCode;
            this.resourceCode = resourceCode;
            this.constraintType = constraintType;
            this.description = description;
        }

        public String getOrderNo() { return orderNo; }
        public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public String getResourceCode() { return resourceCode; }
        public void setResourceCode(String resourceCode) { this.resourceCode = resourceCode; }
        public String getConstraintType() { return constraintType; }
        public void setConstraintType(String constraintType) { this.constraintType = constraintType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }

    public static ScheduleResult success(List<ScheduleOrder> orders, ScheduleDirection direction, long computationTimeMs) {
        ScheduleResult result = new ScheduleResult();
        result.status = ScheduleStatus.SUCCESS;
        result.direction = direction;
        result.scheduledOrders = orders;
        result.computedAt = LocalDateTime.now();
        result.computationTimeMs = computationTimeMs;
        result.totalOrders = orders.size();
        result.scheduledCount = (int) orders.stream().filter(o -> o.getStatus() == ScheduleOrder.ScheduleStatus.SCHEDULED).count();
        result.failedCount = result.totalOrders - result.scheduledCount;
        return result;
    }

    public static ScheduleResult partial(List<ScheduleOrder> orders, List<ScheduleConflict> conflicts, ScheduleDirection direction, long computationTimeMs) {
        ScheduleResult result = new ScheduleResult();
        result.status = ScheduleStatus.PARTIAL;
        result.direction = direction;
        result.scheduledOrders = orders;
        result.conflicts = conflicts;
        result.computedAt = LocalDateTime.now();
        result.computationTimeMs = computationTimeMs;
        result.totalOrders = orders.size() + (int) conflicts.stream().map(ScheduleConflict::getOrderNo).distinct().count();
        result.scheduledCount = (int) orders.stream().filter(o -> o.getStatus() == ScheduleOrder.ScheduleStatus.SCHEDULED).count();
        result.failedCount = result.totalOrders - result.scheduledCount;
        return result;
    }

    public static ScheduleResult failed(List<ScheduleConflict> conflicts, long computationTimeMs) {
        ScheduleResult result = new ScheduleResult();
        result.status = ScheduleStatus.FAILED;
        result.conflicts = conflicts;
        result.computedAt = LocalDateTime.now();
        result.computationTimeMs = computationTimeMs;
        result.totalOrders = 0;
        result.scheduledCount = 0;
        result.failedCount = conflicts.size();
        return result;
    }

    public ScheduleStatus getStatus() { return status; }
    public ScheduleDirection getDirection() { return direction; }
    public List<ScheduleOrder> getScheduledOrders() { return scheduledOrders; }
    public List<ScheduleConflict> getConflicts() { return conflicts; }
    public LocalDateTime getComputedAt() { return computedAt; }
    public long getComputationTimeMs() { return computationTimeMs; }
    public int getTotalOrders() { return totalOrders; }
    public int getScheduledCount() { return scheduledCount; }
    public int getFailedCount() { return failedCount; }
}
