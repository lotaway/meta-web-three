package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class WorkOrder {
    private Long id;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private Integer completedQuantity;
    private WorkOrderStatus status;
    private Priority priority;
    private String workshopId;
    private String processRouteId;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum WorkOrderStatus {
        DRAFT, RELEASED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED
    }

    public enum Priority {
        LOW, NORMAL, HIGH, URGENT
    }

    public void create(String workOrderNo, String productCode, String productName, 
                      Integer quantity, String workshopId, String processRouteId) {
        this.workOrderNo = workOrderNo;
        this.productCode = productCode;
        this.productName = productName;
        this.quantity = quantity;
        this.workshopId = workshopId;
        this.processRouteId = processRouteId;
        this.completedQuantity = 0;
        this.status = WorkOrderStatus.DRAFT;
        this.priority = Priority.NORMAL;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void release() {
        if (status != WorkOrderStatus.DRAFT) {
            throw new IllegalStateException("Can only release DRAFT work orders");
        }
        this.status = WorkOrderStatus.RELEASED;
        this.updatedAt = LocalDateTime.now();
    }

    public void start() {
        if (status != WorkOrderStatus.RELEASED) {
            throw new IllegalStateException("Can only start RELEASED work orders");
        }
        this.status = WorkOrderStatus.IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void pause() {
        if (status != WorkOrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only pause IN_PROGRESS work orders");
        }
        this.status = WorkOrderStatus.PAUSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void resume() {
        if (status != WorkOrderStatus.PAUSED) {
            throw new IllegalStateException("Can only resume PAUSED work orders");
        }
        this.status = WorkOrderStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }

    public void complete() {
        if (status != WorkOrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only complete IN_PROGRESS work orders");
        }
        if (completedQuantity < quantity) {
            throw new IllegalStateException("Cannot complete: not all quantities finished");
        }
        this.status = WorkOrderStatus.COMPLETED;
        this.actualEndTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status == WorkOrderStatus.COMPLETED) {
            throw new IllegalStateException("Cannot cancel completed work orders");
        }
        this.status = WorkOrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateProgress(Integer quantity) {
        if (status != WorkOrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Work order is not in progress");
        }
        this.completedQuantity += quantity;
        if (this.completedQuantity >= this.quantity) {
            this.completedQuantity = this.quantity;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public Double getCompletionRate() {
        if (quantity == 0) return 0.0;
        return (double) completedQuantity / quantity * 100;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public Integer getCompletedQuantity() { return completedQuantity; }
    public void setCompletedQuantity(Integer completedQuantity) { this.completedQuantity = completedQuantity; }
    public WorkOrderStatus getStatus() { return status; }
    public void setStatus(WorkOrderStatus status) { this.status = status; }
    public Priority getPriority() { return priority; }
    public void setPriority(Priority priority) { this.priority = priority; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}