package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class QcPlanItem {
    
    private Long id;
    private Long planId;
    private Long itemId;
    private Integer itemSequence;
    private Boolean isMandatory;
    private String defaultValue;
    private String inspectionMethod;
    private String samplingRule;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private QcInspectionItem inspectionItem;
    
    public static QcPlanItem create(Long planId, Long itemId, Integer itemSequence) {
        QcPlanItem planItem = new QcPlanItem();
        planItem.planId = planId;
        planItem.itemId = itemId;
        planItem.itemSequence = itemSequence;
        planItem.isMandatory = true;
        planItem.sortOrder = itemSequence;
        planItem.createdAt = LocalDateTime.now();
        planItem.updatedAt = LocalDateTime.now();
        return planItem;
    }
    
    public void update(Integer itemSequence, Boolean isMandatory, String defaultValue,
            String inspectionMethod, String samplingRule) {
        this.itemSequence = itemSequence;
        this.isMandatory = isMandatory;
        this.defaultValue = defaultValue;
        this.inspectionMethod = inspectionMethod;
        this.samplingRule = samplingRule;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getPlanId() { return planId; }
    public void setPlanId(Long planId) { this.planId = planId; }
    public Long getItemId() { return itemId; }
    public void setItemId(Long itemId) { this.itemId = itemId; }
    public Integer getItemSequence() { return itemSequence; }
    public void setItemSequence(Integer itemSequence) { this.itemSequence = itemSequence; }
    public Boolean getIsMandatory() { return isMandatory; }
    public void setIsMandatory(Boolean isMandatory) { this.isMandatory = isMandatory; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public String getInspectionMethod() { return inspectionMethod; }
    public void setInspectionMethod(String inspectionMethod) { this.inspectionMethod = inspectionMethod; }
    public String getSamplingRule() { return samplingRule; }
    public void setSamplingRule(String samplingRule) { this.samplingRule = samplingRule; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public QcInspectionItem getInspectionItem() { return inspectionItem; }
    public void setInspectionItem(QcInspectionItem inspectionItem) { this.inspectionItem = inspectionItem; }
}