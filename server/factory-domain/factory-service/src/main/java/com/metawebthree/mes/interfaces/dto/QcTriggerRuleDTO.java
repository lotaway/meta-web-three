package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.QcTriggerRule;
import com.metawebthree.mes.domain.entity.QcTriggerRule.TriggerCondition;
import com.metawebthree.mes.domain.entity.QcTriggerRule.TriggerType;

import java.time.LocalDateTime;

public class QcTriggerRuleDTO {
    
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String triggerType;
    private String targetObject;
    private TriggerConditionDTO condition;
    private String inspectionType;
    private String inspectionPlanCode;
    private Boolean isEnabled;
    private Integer priority;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static QcTriggerRuleDTO fromEntity(QcTriggerRule entity) {
        if (entity == null) return null;
        
        QcTriggerRuleDTO dto = new QcTriggerRuleDTO();
        dto.setId(entity.getId());
        dto.setRuleCode(entity.getRuleCode());
        dto.setRuleName(entity.getRuleName());
        dto.setTriggerType(entity.getTriggerType() != null ? entity.getTriggerType().name() : null);
        dto.setTargetObject(entity.getTargetObject());
        if (entity.getCondition() != null) {
            TriggerConditionDTO conditionDto = new TriggerConditionDTO();
            conditionDto.setBatchSize(entity.getCondition().getBatchSize());
            conditionDto.setTimeIntervalMinutes(entity.getCondition().getTimeIntervalMinutes());
            conditionDto.setQuantityThreshold(entity.getCondition().getQuantityThreshold());
            conditionDto.setEventType(entity.getCondition().getEventType());
            conditionDto.setCronExpression(entity.getCondition().getCronExpression());
            dto.setCondition(conditionDto);
        }
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionPlanCode(entity.getInspectionPlanCode());
        dto.setIsEnabled(entity.getIsEnabled());
        dto.setPriority(entity.getPriority());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        return dto;
    }
    
    public QcTriggerRule toEntity() {
        QcTriggerRule triggerRule = new QcTriggerRule();
        if (this.id != null) {
            triggerRule.setId(this.id);
        }
        triggerRule.setRuleCode(this.ruleCode);
        triggerRule.setRuleName(this.ruleName);
        if (this.triggerType != null) {
            triggerRule.setTriggerType(TriggerType.valueOf(this.triggerType));
        }
        triggerRule.setTargetObject(this.targetObject);
        if (this.condition != null) {
            TriggerCondition cond = new TriggerCondition();
            cond.setBatchSize(this.condition.getBatchSize());
            cond.setTimeIntervalMinutes(this.condition.getTimeIntervalMinutes());
            cond.setQuantityThreshold(this.condition.getQuantityThreshold());
            cond.setEventType(this.condition.getEventType());
            cond.setCronExpression(this.condition.getCronExpression());
            triggerRule.setCondition(cond);
        }
        triggerRule.setInspectionType(this.inspectionType);
        triggerRule.setInspectionPlanCode(this.inspectionPlanCode);
        triggerRule.setIsEnabled(this.isEnabled);
        triggerRule.setPriority(this.priority);
        
        return triggerRule;
    }
    
    public static class TriggerConditionDTO {
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
    
    // ========== Getters and Setters ==========
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public String getTriggerType() { return triggerType; }
    public void setTriggerType(String triggerType) { this.triggerType = triggerType; }
    public String getTargetObject() { return targetObject; }
    public void setTargetObject(String targetObject) { this.targetObject = targetObject; }
    public TriggerConditionDTO getCondition() { return condition; }
    public void setCondition(TriggerConditionDTO condition) { this.condition = condition; }
    public String getInspectionType() { return inspectionType; }
    public void setInspectionType(String inspectionType) { this.inspectionType = inspectionType; }
    public String getInspectionPlanCode() { return inspectionPlanCode; }
    public void setInspectionPlanCode(String inspectionPlanCode) { this.inspectionPlanCode = inspectionPlanCode; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}