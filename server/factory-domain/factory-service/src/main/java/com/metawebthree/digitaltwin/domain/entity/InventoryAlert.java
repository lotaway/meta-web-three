package com.metawebthree.digitaltwin.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import lombok.Data;

@Data
public class InventoryAlert {
    private Long id;
    private String alertCode;
    private String warehouseCode;
    private String shelfCode;
    private String itemCode;
    private AlertType alertType;
    private AlertLevel alertLevel;
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
        this.alertLevel = level;
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

    public boolean isAcknowledged() {
        return status == AlertStatus.ACKNOWLEDGED;
    }

    public Long getResolutionTimeMinutes() {
        if (acknowledgedAt == null || resolvedAt == null) return null;
        return java.time.Duration.between(acknowledgedAt, resolvedAt).toMinutes();
    }

    public boolean shouldAutoEscalate() {
        return alertLevel == AlertLevel.CRITICAL
                || (alertLevel == AlertLevel.ERROR && status == AlertStatus.TRIGGERED);
    }

    public boolean isOverdue(int hours) {
        return acknowledgedAt != null
                && acknowledgedAt.plusHours(hours).isBefore(LocalDateTime.now());
    }

    public int getSeverityScore() {
        return switch (alertLevel) {
            case CRITICAL -> 4;
            case ERROR -> 3;
            case WARNING -> 2;
            case INFO -> 1;
        };
    }
}