package com.metawebthree.digitaltwin.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class InventoryAlert {
    private Long id;
    private String alertCode;
    private String warehouseCode;
    private String shelfCode;
    private String itemCode;
    private AlertType alertType;
    private AlertLevel level;
    private String title;
    private String description;
    private BigDecimal currentQuantity;
    private BigDecimal thresholdValue;
    private AlertStatus status;
    private String solution;
    private String acknowledgedBy;
    private String resolvedBy;
    private LocalDateTime occurredAt;
    private LocalDateTime acknowledgedAt;
    private LocalDateTime resolvedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum AlertType {
        LOW_STOCK, OVERSTOCK, EXPIRING_SOON, EXPIRED, SHELF_FULL, SHELF_EMPTY
    }

    public enum AlertLevel {
        INFO, WARNING, ERROR, CRITICAL
    }

    public enum AlertStatus {
        TRIGGERED, ACKNOWLEDGED, IN_PROGRESS, RESOLVED, CLOSED
    }

    public InventoryAlert() {
    }

    public InventoryAlert(String alertCode, String itemCode, AlertType alertType, AlertLevel level, String title) {
        this.alertCode = alertCode;
        this.itemCode = itemCode;
        this.alertType = alertType;
        this.level = level;
        this.title = title;
        this.status = AlertStatus.TRIGGERED;
        this.occurredAt = LocalDateTime.now();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void acknowledge(String userId) {
        this.status = AlertStatus.ACKNOWLEDGED;
        this.acknowledgedBy = userId;
        this.acknowledgedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void startResolution() {
        this.status = AlertStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }

    public void resolve(String userId, String solution) {
        this.status = AlertStatus.RESOLVED;
        this.resolvedBy = userId;
        this.resolvedAt = LocalDateTime.now();
        this.solution = solution;
        this.updatedAt = LocalDateTime.now();
    }

    public void close() {
        this.status = AlertStatus.CLOSED;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isActive() {
        return status == AlertStatus.TRIGGERED
                || status == AlertStatus.ACKNOWLEDGED
                || status == AlertStatus.IN_PROGRESS;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getAlertCode() {
        return alertCode;
    }

    public void setAlertCode(String alertCode) {
        this.alertCode = alertCode;
    }

    public String getWarehouseCode() {
        return warehouseCode;
    }

    public void setWarehouseCode(String warehouseCode) {
        this.warehouseCode = warehouseCode;
    }

    public String getShelfCode() {
        return shelfCode;
    }

    public void setShelfCode(String shelfCode) {
        this.shelfCode = shelfCode;
    }

    public String getItemCode() {
        return itemCode;
    }

    public void setItemCode(String itemCode) {
        this.itemCode = itemCode;
    }

    public AlertType getAlertType() {
        return alertType;
    }

    public void setAlertType(AlertType alertType) {
        this.alertType = alertType;
    }

    public AlertLevel getLevel() {
        return level;
    }

    public void setLevel(AlertLevel level) {
        this.level = level;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public BigDecimal getCurrentQuantity() {
        return currentQuantity;
    }

    public void setCurrentQuantity(BigDecimal currentQuantity) {
        this.currentQuantity = currentQuantity;
    }

    public BigDecimal getThresholdValue() {
        return thresholdValue;
    }

    public void setThresholdValue(BigDecimal thresholdValue) {
        this.thresholdValue = thresholdValue;
    }

    public AlertStatus getStatus() {
        return status;
    }

    public void setStatus(AlertStatus status) {
        this.status = status;
    }

    public String getSolution() {
        return solution;
    }

    public void setSolution(String solution) {
        this.solution = solution;
    }

    public String getAcknowledgedBy() {
        return acknowledgedBy;
    }

    public void setAcknowledgedBy(String acknowledgedBy) {
        this.acknowledgedBy = acknowledgedBy;
    }

    public String getResolvedBy() {
        return resolvedBy;
    }

    public void setResolvedBy(String resolvedBy) {
        this.resolvedBy = resolvedBy;
    }

    public LocalDateTime getOccurredAt() {
        return occurredAt;
    }

    public void setOccurredAt(LocalDateTime occurredAt) {
        this.occurredAt = occurredAt;
    }

    public LocalDateTime getAcknowledgedAt() {
        return acknowledgedAt;
    }

    public void setAcknowledgedAt(LocalDateTime acknowledgedAt) {
        this.acknowledgedAt = acknowledgedAt;
    }

    public LocalDateTime getResolvedAt() {
        return resolvedAt;
    }

    public void setResolvedAt(LocalDateTime resolvedAt) {
        this.resolvedAt = resolvedAt;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}