package com.metawebthree.digitaltwin.domain.entity;

import java.time.LocalDateTime;

public class Alert {
    private Long id;
    private String alertCode;
    private String deviceCode;
    private String workshopId;
    private AlertLevel level;
    private AlertType type;
    private AlertStatus status;
    private String title;
    private String description;
    private String solution;
    private LocalDateTime occurredAt;
    private LocalDateTime acknowledgedAt;
    private LocalDateTime resolvedAt;
    private String acknowledgedBy;
    private String resolvedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum AlertLevel {
        INFO, WARNING, ERROR, CRITICAL
    }

    public enum AlertType {
        DEVICE_OFFLINE, DEVICE_ERROR, TEMPERATURE_HIGH, PRESSURE_ABNORMAL,
        VIBRATION_ABNORMAL, PRODUCTION_STOP, QUALITY_ISSUE, MAINTENANCE_DUE,
        NETWORK_ERROR, POWER_FAILURE, SAFETY_ISSUE, OTHER
    }

    public enum AlertStatus {
        TRIGGERED, ACKNOWLEDGED, IN_PROGRESS, RESOLVED, CLOSED
    }

    public void create(String alertCode, String deviceCode, String workshopId,
                      AlertLevel level, AlertType type, String title, String description) {
        this.alertCode = alertCode;
        this.deviceCode = deviceCode;
        this.workshopId = workshopId;
        this.level = level;
        this.type = type;
        this.title = title;
        this.description = description;
        this.status = AlertStatus.TRIGGERED;
        this.occurredAt = LocalDateTime.now();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void acknowledge(String acknowledgedBy) {
        this.status = AlertStatus.ACKNOWLEDGED;
        this.acknowledgedAt = LocalDateTime.now();
        this.acknowledgedBy = acknowledgedBy;
        this.updatedAt = LocalDateTime.now();
    }

    public void startResolve() {
        this.status = AlertStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }

    public void resolve(String solution, String resolvedBy) {
        this.status = AlertStatus.RESOLVED;
        this.solution = solution;
        this.resolvedAt = LocalDateTime.now();
        this.resolvedBy = resolvedBy;
        this.updatedAt = LocalDateTime.now();
    }

    public void close() {
        this.status = AlertStatus.CLOSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void escalate() {
        if (level == null || level == AlertLevel.CRITICAL) {
            return;
        }
        int ordinal = level.ordinal();
        if (ordinal < AlertLevel.values().length - 1) {
            this.level = AlertLevel.values()[ordinal + 1];
            this.updatedAt = LocalDateTime.now();
        }
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getAlertCode() { return alertCode; }
    public void setAlertCode(String alertCode) { this.alertCode = alertCode; }
    public String getDeviceCode() { return deviceCode; }
    public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public AlertLevel getLevel() { return level; }
    public void setLevel(AlertLevel level) { this.level = level; }
    public AlertType getType() { return type; }
    public void setType(AlertType type) { this.type = type; }
    public AlertStatus getStatus() { return status; }
    public void setStatus(AlertStatus status) { this.status = status; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getSolution() { return solution; }
    public void setSolution(String solution) { this.solution = solution; }
    public LocalDateTime getOccurredAt() { return occurredAt; }
    public void setOccurredAt(LocalDateTime occurredAt) { this.occurredAt = occurredAt; }
    public LocalDateTime getAcknowledgedAt() { return acknowledgedAt; }
    public void setAcknowledgedAt(LocalDateTime acknowledgedAt) { this.acknowledgedAt = acknowledgedAt; }
    public LocalDateTime getResolvedAt() { return resolvedAt; }
    public void setResolvedAt(LocalDateTime resolvedAt) { this.resolvedAt = resolvedAt; }
    public String getAcknowledgedBy() { return acknowledgedBy; }
    public void setAcknowledgedBy(String acknowledgedBy) { this.acknowledgedBy = acknowledgedBy; }
    public String getResolvedBy() { return resolvedBy; }
    public void setResolvedBy(String resolvedBy) { this.resolvedBy = resolvedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    // equals/hashCode based on business key (alertCode)
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Alert alert = (Alert) o;
        return alertCode != null && alertCode.equals(alert.alertCode);
    }

    @Override
    public int hashCode() {
        return alertCode != null ? alertCode.hashCode() : 0;
    }
}