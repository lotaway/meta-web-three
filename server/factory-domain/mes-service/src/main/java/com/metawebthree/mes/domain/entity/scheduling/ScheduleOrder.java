package com.metawebthree.mes.domain.entity.scheduling;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

public class ScheduleOrder {

    public enum ScheduleStatus {
        PENDING, SCHEDULED, IN_PROGRESS, COMPLETED, DELAYED, CANCELLED
    }

    public enum Priority {
        LOW, NORMAL, HIGH, URGENT
    }

    private Long id;
    private String scheduleNo;
    private String orderNo;
    private String productCode;
    private String productName;
    private BigDecimal quantity;
    private BigDecimal completedQuantity;
    private LocalDateTime dueDate;
    private LocalDateTime scheduledStartTime;
    private LocalDateTime scheduledEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private Priority priority;
    private ScheduleStatus status;
    private String workshopId;
    private String routeCode;
    private List<ScheduleOperation> operations;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ScheduleOperationStatus {
        PENDING, SCHEDULED, IN_PROGRESS, COMPLETED, BLOCKED
    }

    public static class ScheduleOperation {
        private Long id;
        private Long scheduleOrderId;
        private String operationCode;
        private String operationName;
        private Integer sequenceNo;
        private String resourceCode;
        private String resourceName;
        private BigDecimal setupTimeMinutes;
        private BigDecimal processingTimeMinutes;
        private BigDecimal teardownTimeMinutes;
        private ScheduleOperationStatus status;
        private LocalDateTime scheduledStartTime;
        private LocalDateTime scheduledEndTime;

        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getScheduleOrderId() { return scheduleOrderId; }
        public void setScheduleOrderId(Long scheduleOrderId) { this.scheduleOrderId = scheduleOrderId; }
        public String getOperationCode() { return operationCode; }
        public void setOperationCode(String operationCode) { this.operationCode = operationCode; }
        public String getOperationName() { return operationName; }
        public void setOperationName(String operationName) { this.operationName = operationName; }
        public Integer getSequenceNo() { return sequenceNo; }
        public void setSequenceNo(Integer sequenceNo) { this.sequenceNo = sequenceNo; }
        public String getResourceCode() { return resourceCode; }
        public void setResourceCode(String resourceCode) { this.resourceCode = resourceCode; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public BigDecimal getSetupTimeMinutes() { return setupTimeMinutes; }
        public void setSetupTimeMinutes(BigDecimal setupTimeMinutes) { this.setupTimeMinutes = setupTimeMinutes; }
        public BigDecimal getProcessingTimeMinutes() { return processingTimeMinutes; }
        public void setProcessingTimeMinutes(BigDecimal processingTimeMinutes) { this.processingTimeMinutes = processingTimeMinutes; }
        public BigDecimal getTeardownTimeMinutes() { return teardownTimeMinutes; }
        public void setTeardownTimeMinutes(BigDecimal teardownTimeMinutes) { this.teardownTimeMinutes = teardownTimeMinutes; }
        public ScheduleOperationStatus getStatus() { return status; }
        public void setStatus(ScheduleOperationStatus status) { this.status = status; }
        public LocalDateTime getScheduledStartTime() { return scheduledStartTime; }
        public void setScheduledStartTime(LocalDateTime scheduledStartTime) { this.scheduledStartTime = scheduledStartTime; }
        public LocalDateTime getScheduledEndTime() { return scheduledEndTime; }
        public void setScheduledEndTime(LocalDateTime scheduledEndTime) { this.scheduledEndTime = scheduledEndTime; }
    }

    public void create(String scheduleNo, String orderNo, String productCode, String productName,
                        BigDecimal quantity, LocalDateTime dueDate, Priority priority, String workshopId, String routeCode) {
        this.scheduleNo = scheduleNo;
        this.orderNo = orderNo;
        this.productCode = productCode;
        this.productName = productName;
        this.quantity = quantity;
        this.completedQuantity = BigDecimal.ZERO;
        this.dueDate = dueDate;
        this.priority = priority;
        this.workshopId = workshopId;
        this.routeCode = routeCode;
        this.status = ScheduleStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void schedule(LocalDateTime startTime, LocalDateTime endTime, List<ScheduleOperation> ops) {
        this.scheduledStartTime = startTime;
        this.scheduledEndTime = endTime;
        this.operations = ops;
        this.status = ScheduleStatus.SCHEDULED;
        this.updatedAt = LocalDateTime.now();
    }

    public void start() {
        this.status = ScheduleStatus.IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void complete() {
        this.status = ScheduleStatus.COMPLETED;
        this.actualEndTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void markDelayed() {
        this.status = ScheduleStatus.DELAYED;
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        this.status = ScheduleStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateCompletedQuantity(BigDecimal qty) {
        this.completedQuantity = qty;
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal getCompletionRate() {
        if (quantity == null || quantity.compareTo(BigDecimal.ZERO) <= 0) return BigDecimal.ZERO;
        return completedQuantity != null
            ? completedQuantity.multiply(BigDecimal.valueOf(100)).divide(quantity, 2, BigDecimal.ROUND_HALF_UP)
            : BigDecimal.ZERO;
    }

    public boolean isOverdue() {
        return dueDate != null && status == ScheduleStatus.PENDING && LocalDateTime.now().isAfter(dueDate);
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getScheduleNo() { return scheduleNo; }
    public void setScheduleNo(String scheduleNo) { this.scheduleNo = scheduleNo; }
    public String getOrderNo() { return orderNo; }
    public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public BigDecimal getQuantity() { return quantity; }
    public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
    public BigDecimal getCompletedQuantity() { return completedQuantity; }
    public void setCompletedQuantity(BigDecimal completedQuantity) { this.completedQuantity = completedQuantity; }
    public LocalDateTime getDueDate() { return dueDate; }
    public void setDueDate(LocalDateTime dueDate) { this.dueDate = dueDate; }
    public LocalDateTime getScheduledStartTime() { return scheduledStartTime; }
    public void setScheduledStartTime(LocalDateTime scheduledStartTime) { this.scheduledStartTime = scheduledStartTime; }
    public LocalDateTime getScheduledEndTime() { return scheduledEndTime; }
    public void setScheduledEndTime(LocalDateTime scheduledEndTime) { this.scheduledEndTime = scheduledEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public Priority getPriority() { return priority; }
    public void setPriority(Priority priority) { this.priority = priority; }
    public ScheduleStatus getStatus() { return status; }
    public void setStatus(ScheduleStatus status) { this.status = status; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getRouteCode() { return routeCode; }
    public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    public List<ScheduleOperation> getOperations() { return operations; }
    public void setOperations(List<ScheduleOperation> operations) { this.operations = operations; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
