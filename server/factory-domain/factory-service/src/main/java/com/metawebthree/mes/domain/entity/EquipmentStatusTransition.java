package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class EquipmentStatusTransition {
    private Long id;
    private Long equipmentTypeId;
    private String fromStatusCode;
    private String toStatusCode;
    private String transitionAction;
    private String conditionExpression;
    private String eventCode;
    private Boolean isAutoTransition;
    private Integer timeoutSeconds;
    private String timeoutAction;
    private String description;
    private Integer sortOrder;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long equipmentTypeId, String fromStatusCode, String toStatusCode, String transitionAction) {
        this.equipmentTypeId = equipmentTypeId;
        this.fromStatusCode = fromStatusCode;
        this.toStatusCode = toStatusCode;
        this.transitionAction = transitionAction;
        this.isAutoTransition = false;
        this.isActive = true;
        this.sortOrder = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setCondition(String conditionExpression) {
        this.conditionExpression = conditionExpression;
        this.updatedAt = LocalDateTime.now();
    }

    public void bindEvent(String eventCode) {
        this.eventCode = eventCode;
        this.updatedAt = LocalDateTime.now();
    }

    public void enableAutoTransition(Integer timeoutSeconds, String timeoutAction) {
        this.isAutoTransition = true;
        this.timeoutSeconds = timeoutSeconds;
        this.timeoutAction = timeoutAction;
        this.updatedAt = LocalDateTime.now();
    }

    public void disableAutoTransition() {
        this.isAutoTransition = false;
        this.timeoutSeconds = null;
        this.timeoutAction = null;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean canTransition(String currentStatus, String triggerEvent) {
        if (!isActive) {
            return false;
        }
        if (!fromStatusCode.equals(currentStatus)) {
            return false;
        }
        if (eventCode != null && !eventCode.equals(triggerEvent)) {
            return false;
        }
        return true;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getEquipmentTypeId() { return equipmentTypeId; }
    public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
    public String getFromStatusCode() { return fromStatusCode; }
    public void setFromStatusCode(String fromStatusCode) { this.fromStatusCode = fromStatusCode; }
    public String getToStatusCode() { return toStatusCode; }
    public void setToStatusCode(String toStatusCode) { this.toStatusCode = toStatusCode; }
    public String getTransitionAction() { return transitionAction; }
    public void setTransitionAction(String transitionAction) { this.transitionAction = transitionAction; }
    public String getConditionExpression() { return conditionExpression; }
    public void setConditionExpression(String conditionExpression) { this.conditionExpression = conditionExpression; }
    public String getEventCode() { return eventCode; }
    public void setEventCode(String eventCode) { this.eventCode = eventCode; }
    public Boolean getIsAutoTransition() { return isAutoTransition; }
    public void setIsAutoTransition(Boolean isAutoTransition) { this.isAutoTransition = isAutoTransition; }
    public Integer getTimeoutSeconds() { return timeoutSeconds; }
    public void setTimeoutSeconds(Integer timeoutSeconds) { this.timeoutSeconds = timeoutSeconds; }
    public String getTimeoutAction() { return timeoutAction; }
    public void setTimeoutAction(String timeoutAction) { this.timeoutAction = timeoutAction; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}