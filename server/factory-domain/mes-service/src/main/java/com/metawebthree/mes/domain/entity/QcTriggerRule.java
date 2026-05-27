package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

import com.metawebthree.mes.domain.QcConstants;

public class QcTriggerRule {
    private Long id;
    private String ruleCode;
    private String ruleName;
    private TriggerType triggerType;
    private String targetObject;
    private TriggerCondition condition;
    private String inspectionType;
    private String inspectionPlanCode;
    private Boolean isEnabled;
    private Integer priority;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum TriggerType {
        BY_BATCH, BY_TIME, BY_QUANTITY, BY_EVENT, MANUAL
    }

    public static class TriggerCondition {
        private Integer batchSize;
        private Integer timeIntervalMinutes;
        private Integer quantityThreshold;
        private String eventType;
        private String cronExpression;

        public Integer getBatchSize() { return batchSize; }
        public void setBatchSize(Integer batchSize) { this.batchSize = batchSize; }
        public Integer getTimeIntervalMinutes() { return timeIntervalMinutes; }
        public void setTimeIntervalMinutes(Integer timeIntervalMinutes) { this.timeIntervalMinutes = timeIntervalMinutes; }
        public Integer getQuantityThreshold() { return quantityThreshold; }
        public void setQuantityThreshold(Integer quantityThreshold) { this.quantityThreshold = quantityThreshold; }
        public String getEventType() { return eventType; }
        public void setEventType(String eventType) { this.eventType = eventType; }
        public String getCronExpression() { return cronExpression; }
        public void setCronExpression(String cronExpression) { this.cronExpression = cronExpression; }
    }

    public void create(String ruleCode, String ruleName, TriggerType triggerType, 
                       String targetObject) {
        this.ruleCode = ruleCode;
        this.ruleName = ruleName;
        this.triggerType = triggerType;
        this.targetObject = targetObject;
        this.condition = new TriggerCondition();
        this.isEnabled = Boolean.TRUE;
        this.priority = QcConstants.DEFAULT_SORT_ORDER;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setBatchTrigger(Integer batchSize) {
        this.triggerType = TriggerType.BY_BATCH;
        if (this.condition == null) {
            this.condition = new TriggerCondition();
        }
        this.condition.setBatchSize(batchSize);
        this.updatedAt = LocalDateTime.now();
    }

    public void setTimeTrigger(Integer intervalMinutes) {
        this.triggerType = TriggerType.BY_TIME;
        if (this.condition == null) {
            this.condition = new TriggerCondition();
        }
        this.condition.setTimeIntervalMinutes(intervalMinutes);
        this.updatedAt = LocalDateTime.now();
    }

    public void setQuantityTrigger(Integer threshold) {
        this.triggerType = TriggerType.BY_QUANTITY;
        if (this.condition == null) {
            this.condition = new TriggerCondition();
        }
        this.condition.setQuantityThreshold(threshold);
        this.updatedAt = LocalDateTime.now();
    }

    public void setEventTrigger(String eventType) {
        this.triggerType = TriggerType.BY_EVENT;
        if (this.condition == null) {
            this.condition = new TriggerCondition();
        }
        this.condition.setEventType(eventType);
        this.updatedAt = LocalDateTime.now();
    }

    public void setCronTrigger(String cronExpression) {
        this.triggerType = TriggerType.BY_TIME;
        if (this.condition == null) {
            this.condition = new TriggerCondition();
        }
        this.condition.setCronExpression(cronExpression);
        this.updatedAt = LocalDateTime.now();
    }

    public void bindInspectionPlan(String inspectionType, String inspectionPlanCode) {
        this.inspectionType = inspectionType;
        this.inspectionPlanCode = inspectionPlanCode;
        this.updatedAt = LocalDateTime.now();
    }

    public void updatePriority(Integer priority) {
        this.priority = priority;
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        this.isEnabled = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.isEnabled = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    public Boolean shouldTrigger(Integer currentBatchSize, Integer currentQuantity) {
        if (!Boolean.TRUE.equals(this.isEnabled)) {
            return Boolean.FALSE;
        }
        return switch (this.triggerType) {
            case BY_BATCH -> currentBatchSize != null && 
                             this.condition.getBatchSize() != null && 
                             currentBatchSize >= this.condition.getBatchSize();
            case BY_QUANTITY -> currentQuantity != null && 
                                this.condition.getQuantityThreshold() != null && 
                                currentQuantity >= this.condition.getQuantityThreshold();
            default -> Boolean.FALSE;
        };
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public TriggerType getTriggerType() { return triggerType; }
    public void setTriggerType(TriggerType triggerType) { this.triggerType = triggerType; }
    public String getTargetObject() { return targetObject; }
    public void setTargetObject(String targetObject) { this.targetObject = targetObject; }
    public TriggerCondition getCondition() { return condition; }
    public void setCondition(TriggerCondition condition) { this.condition = condition; }
    public String getInspectionType() { return inspectionType; }
    public void setInspectionType(String inspectionType) { this.inspectionType = inspectionType; }
    public String getInspectionPlanCode() { return inspectionPlanCode; }
    public void setInspectionPlanCode(String inspectionPlanCode) { this.inspectionPlanCode = inspectionPlanCode; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}