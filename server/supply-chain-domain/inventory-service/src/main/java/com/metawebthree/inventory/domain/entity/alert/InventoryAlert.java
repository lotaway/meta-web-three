package com.metawebthree.inventory.domain.entity.alert;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryAlert {
    private Long id;
    private String alertCode;
    private String warehouseCode;
    private String skuCode;
    private AlertType alertType;
    private AlertLevel level;
    private String title;
    private String description;
    private Integer currentQuantity;
    private Integer thresholdValue;
    private AlertStatus status;
    private String solution;
    private String acknowledgedBy;
    private String resolvedBy;
    private LocalDateTime occurredAt;
    private LocalDateTime acknowledgedAt;
    private LocalDateTime resolvedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public enum AlertType {
        LOW_STOCK, OVERSTOCK, EXPIRING_SOON, EXPIRED
    }

    public enum AlertLevel {
        INFO, WARNING, ERROR, CRITICAL
    }

    public enum AlertStatus {
        TRIGGERED, ACKNOWLEDGED, IN_PROGRESS, RESOLVED, CLOSED
    }

    public InventoryAlert() {
    }

    public InventoryAlert(String alertCode, String warehouseCode, String skuCode, 
                          AlertType alertType, AlertLevel level, String title,
                          Integer currentQuantity, Integer thresholdValue) {
        this.alertCode = alertCode;
        this.warehouseCode = warehouseCode;
        this.skuCode = skuCode;
        this.alertType = alertType;
        this.level = level;
        this.title = title;
        this.currentQuantity = currentQuantity;
        this.thresholdValue = thresholdValue;
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
}