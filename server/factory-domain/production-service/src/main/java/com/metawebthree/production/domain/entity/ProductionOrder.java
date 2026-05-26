package com.metawebthree.production.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class ProductionOrder {
    private Long id;
    private String orderCode;
    private String productCode;
    private String productName;
    private Integer quantityPlanned;
    private Integer quantityCompleted;
    private OrderStatus status;
    private Priority priority;
    private String workshopCode;
    private String productionLineCode;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private Double progressPercentage;
    private String orderType;
    private String customerName;
    private String notes;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<ProductionSchedule> schedules;

    public enum OrderStatus {
        PENDING, SCHEDULED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED
    }

    public enum Priority {
        LOW, NORMAL, HIGH, URGENT
    }

    public ProductionOrder() {
        this.status = OrderStatus.PENDING;
        this.progressPercentage = 0.0;
        this.quantityCompleted = 0;
        this.schedules = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateProgress() {
        if (quantityPlanned > 0) {
            this.progressPercentage = (double) quantityCompleted / quantityPlanned * 100;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void startProduction() {
        if (this.status != OrderStatus.SCHEDULED) {
            throw new IllegalStateException("Can only start scheduled orders");
        }
        this.status = OrderStatus.IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void pauseProduction() {
        if (this.status != OrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only pause orders in progress");
        }
        this.status = OrderStatus.PAUSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void resumeProduction() {
        if (this.status != OrderStatus.PAUSED) {
            throw new IllegalStateException("Can only resume paused orders");
        }
        this.status = OrderStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeProduction() {
        if (this.status != OrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only complete orders in progress");
        }
        this.status = OrderStatus.COMPLETED;
        this.quantityCompleted = this.quantityPlanned;
        this.progressPercentage = 100.0;
        this.actualEndTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void cancelOrder() {
        if (this.status == OrderStatus.COMPLETED) {
            throw new IllegalStateException("Cannot cancel completed orders");
        }
        this.status = OrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isOverdue() {
        if (this.plannedEndTime == null) return false;
        return LocalDateTime.now().isAfter(this.plannedEndTime) 
            && this.status != OrderStatus.COMPLETED 
            && this.status != OrderStatus.CANCELLED;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getOrderCode() { return orderCode; }
    public void setOrderCode(String orderCode) { this.orderCode = orderCode; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public Integer getQuantityPlanned() { return quantityPlanned; }
    public void setQuantityPlanned(Integer quantityPlanned) { this.quantityPlanned = quantityPlanned; }
    public Integer getQuantityCompleted() { return quantityCompleted; }
    public void setQuantityCompleted(Integer quantityCompleted) { this.quantityCompleted = quantityCompleted; }
    public OrderStatus getStatus() { return status; }
    public void setStatus(OrderStatus status) { this.status = status; }
    public Priority getPriority() { return priority; }
    public void setPriority(Priority priority) { this.priority = priority; }
    public String getWorkshopCode() { return workshopCode; }
    public void setWorkshopCode(String workshopCode) { this.workshopCode = workshopCode; }
    public String getProductionLineCode() { return productionLineCode; }
    public void setProductionLineCode(String productionLineCode) { this.productionLineCode = productionLineCode; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public Double getProgressPercentage() { return progressPercentage; }
    public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
    public String getOrderType() { return orderType; }
    public void setOrderType(String orderType) { this.orderType = orderType; }
    public String getCustomerName() { return customerName; }
    public void setCustomerName(String customerName) { this.customerName = customerName; }
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public List<ProductionSchedule> getSchedules() { return schedules; }
    public void setSchedules(List<ProductionSchedule> schedules) { this.schedules = schedules; }
}