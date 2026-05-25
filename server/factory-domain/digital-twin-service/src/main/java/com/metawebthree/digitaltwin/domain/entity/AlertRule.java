package com.metawebthree.digitaltwin.domain.entity;

import java.time.LocalDateTime;
import java.util.Objects;

public class AlertRule {
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String description;
    private String deviceType;
    private String deviceCode;
    private String workshopId;
    private MetricType metricType;
    private ComparisonOperator operator;
    private Double thresholdValue;
    private Integer durationSeconds;
    private AlertRuleLevel level;
    private AlertType alertType;
    private String titleTemplate;
    private String descriptionTemplate;
    private Boolean enabled;
    private Integer cooldownSeconds;
    private Integer maxAlertsPerHour;
    private String notificationChannels;
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;

    public enum MetricType {
        TEMPERATURE, HUMIDITY, PRESSURE, VIBRATION, POWER, RPM,
        PRODUCTION_RATE, DEFECT_RATE, RESPONSE_TIME, CUSTOM
    }

    public enum ComparisonOperator {
        GREATER_THAN, LESS_THAN, GREATER_OR_EQUAL, LESS_OR_EQUAL,
        EQUAL, NOT_EQUAL, BETWEEN, OUTSIDE
    }

    public enum AlertRuleLevel {
        INFO, WARNING, ERROR, CRITICAL
    }

    public enum AlertType {
        DEVICE_OFFLINE, DEVICE_ERROR, TEMPERATURE_HIGH, TEMPERATURE_LOW,
        HUMIDITY_HIGH, HUMIDITY_LOW, PRESSURE_ABNORMAL, VIBRATION_ABNORMAL,
        POWER_ABNORMAL, NETWORK_ERROR, MAINTENANCE_DUE, SAFETY_ISSUE, OTHER
    }

    public void createRule(String ruleCode, String ruleName, String description,
                          String deviceType, MetricType metricType, ComparisonOperator operator,
                          Double thresholdValue, AlertRuleLevel level, AlertType alertType,
                          String titleTemplate, String descriptionTemplate, String createdBy) {
        this.ruleCode = ruleCode;
        this.ruleName = ruleName;
        this.description = description;
        this.deviceType = deviceType;
        this.metricType = metricType;
        this.operator = operator;
        this.thresholdValue = thresholdValue;
        this.level = level;
        this.alertType = alertType;
        this.titleTemplate = titleTemplate;
        this.descriptionTemplate = descriptionTemplate;
        this.enabled = true;
        this.cooldownSeconds = 300;
        this.maxAlertsPerHour = 10;
        this.createdBy = createdBy;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public boolean evaluate(Double currentValue, Double secondaryValue) {
        if (!Boolean.TRUE.equals(enabled)) {
            return false;
        }
        return switch (operator) {
            case GREATER_THAN -> currentValue > thresholdValue;
            case LESS_THAN -> currentValue < thresholdValue;
            case GREATER_OR_EQUAL -> currentValue >= thresholdValue;
            case LESS_OR_EQUAL -> currentValue <= thresholdValue;
            case EQUAL -> Objects.equals(currentValue, thresholdValue);
            case NOT_EQUAL -> !Objects.equals(currentValue, thresholdValue);
            case BETWEEN -> secondaryValue != null && 
                           currentValue >= thresholdValue && currentValue <= secondaryValue;
            case OUTSIDE -> secondaryValue != null && 
                          (currentValue < thresholdValue || currentValue > secondaryValue);
        };
    }

    public String generateTitle(String deviceCode, Double currentValue) {
        return titleTemplate != null ? 
            titleTemplate.replace("{device}", deviceCode).replace("{value}", String.valueOf(currentValue)) :
            ruleName;
    }

    public String generateDescription(Double currentValue) {
        return descriptionTemplate != null ?
            descriptionTemplate.replace("{value}", String.valueOf(currentValue))
                              .replace("{threshold}", String.valueOf(thresholdValue)) :
            description;
    }

    public void enable() {
        this.enabled = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        this.enabled = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void update(String ruleName, String description, String deviceType,
                      MetricType metricType, ComparisonOperator operator, Double thresholdValue,
                      Integer durationSeconds, AlertRuleLevel level, AlertType alertType,
                      String titleTemplate, String descriptionTemplate, Integer cooldownSeconds,
                      Integer maxAlertsPerHour, String notificationChannels, String updatedBy) {
        this.ruleName = ruleName;
        this.description = description;
        this.deviceType = deviceType;
        this.metricType = metricType;
        this.operator = operator;
        this.thresholdValue = thresholdValue;
        this.durationSeconds = durationSeconds;
        this.level = level;
        this.alertType = alertType;
        this.titleTemplate = titleTemplate;
        this.descriptionTemplate = descriptionTemplate;
        this.cooldownSeconds = cooldownSeconds;
        this.maxAlertsPerHour = maxAlertsPerHour;
        this.notificationChannels = notificationChannels;
        this.updatedBy = updatedBy;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getDeviceType() { return deviceType; }
    public void setDeviceType(String deviceType) { this.deviceType = deviceType; }
    public String getDeviceCode() { return deviceCode; }
    public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public MetricType getMetricType() { return metricType; }
    public void setMetricType(MetricType metricType) { this.metricType = metricType; }
    public ComparisonOperator getOperator() { return operator; }
    public void setOperator(ComparisonOperator operator) { this.operator = operator; }
    public Double getThresholdValue() { return thresholdValue; }
    public void setThresholdValue(Double thresholdValue) { this.thresholdValue = thresholdValue; }
    public Integer getDurationSeconds() { return durationSeconds; }
    public void setDurationSeconds(Integer durationSeconds) { this.durationSeconds = durationSeconds; }
    public AlertRuleLevel getLevel() { return level; }
    public void setLevel(AlertRuleLevel level) { this.level = level; }
    public AlertType getAlertType() { return alertType; }
    public void setAlertType(AlertType alertType) { this.alertType = alertType; }
    public String getTitleTemplate() { return titleTemplate; }
    public void setTitleTemplate(String titleTemplate) { this.titleTemplate = titleTemplate; }
    public String getDescriptionTemplate() { return descriptionTemplate; }
    public void setDescriptionTemplate(String descriptionTemplate) { this.descriptionTemplate = descriptionTemplate; }
    public Boolean getEnabled() { return enabled; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public Integer getCooldownSeconds() { return cooldownSeconds; }
    public void setCooldownSeconds(Integer cooldownSeconds) { this.cooldownSeconds = cooldownSeconds; }
    public Integer getMaxAlertsPerHour() { return maxAlertsPerHour; }
    public void setMaxAlertsPerHour(Integer maxAlertsPerHour) { this.maxAlertsPerHour = maxAlertsPerHour; }
    public String getNotificationChannels() { return notificationChannels; }
    public void setNotificationChannels(String notificationChannels) { this.notificationChannels = notificationChannels; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}