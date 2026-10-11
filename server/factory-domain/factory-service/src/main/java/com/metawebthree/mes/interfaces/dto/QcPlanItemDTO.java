package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.QcPlanItem;

public class QcPlanItemDTO {
    
    private Long id;
    private Long planId;
    private Long itemId;
    private Integer itemSequence;
    private Boolean isMandatory;
    private String defaultValue;
    private String inspectionMethod;
    private String samplingRule;
    private Integer sortOrder;
    
    private QcInspectionItemDTO inspectionItem;
    
    public static QcPlanItemDTO fromEntity(QcPlanItem entity) {
        if (entity == null) return null;
        
        QcPlanItemDTO dto = new QcPlanItemDTO();
        dto.setId(entity.getId());
        dto.setPlanId(entity.getPlanId());
        dto.setItemId(entity.getItemId());
        dto.setItemSequence(entity.getItemSequence());
        dto.setIsMandatory(entity.getIsMandatory());
        dto.setDefaultValue(entity.getDefaultValue());
        dto.setInspectionMethod(entity.getInspectionMethod());
        dto.setSamplingRule(entity.getSamplingRule());
        dto.setSortOrder(entity.getSortOrder());
        if (entity.getInspectionItem() != null) {
            dto.setInspectionItem(QcInspectionItemDTO.fromEntity(entity.getInspectionItem()));
        }
        return dto;
    }
    
    public QcPlanItem toEntity() {
        QcPlanItem entity = new QcPlanItem();
        entity.setId(this.id);
        entity.setPlanId(this.planId);
        entity.setItemId(this.itemId);
        entity.setItemSequence(this.itemSequence);
        entity.setIsMandatory(this.isMandatory);
        entity.setDefaultValue(this.defaultValue);
        entity.setInspectionMethod(this.inspectionMethod);
        entity.setSamplingRule(this.samplingRule);
        entity.setSortOrder(this.sortOrder);
        return entity;
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
    public QcInspectionItemDTO getInspectionItem() { return inspectionItem; }
    public void setInspectionItem(QcInspectionItemDTO inspectionItem) { this.inspectionItem = inspectionItem; }
}