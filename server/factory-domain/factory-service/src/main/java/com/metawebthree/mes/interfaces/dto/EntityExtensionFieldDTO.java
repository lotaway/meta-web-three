package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;

public class EntityExtensionFieldDTO {
    
    private Long id;
    private String entityType;
    private String fieldCode;
    private String fieldName;
    private String fieldType;
    private String defaultValue;
    private Boolean required;
    private Boolean isUnique;
    private String validationRule;
    private Boolean listVisible;
    private Boolean searchable;
    private Integer sortOrder;
    private String fieldGroup;
    private String referenceType;
    private String referenceEntity;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static EntityExtensionFieldDTO fromEntity(
            com.metawebthree.mes.domain.entity.EntityExtensionField entity) {
        if (entity == null) return null;
        
        EntityExtensionFieldDTO dto = new EntityExtensionFieldDTO();
        dto.setId(entity.getId());
        dto.setEntityType(entity.getEntityType());
        dto.setFieldCode(entity.getFieldCode());
        dto.setFieldName(entity.getFieldName());
        dto.setFieldType(entity.getFieldType() != null ? entity.getFieldType().name() : null);
        dto.setDefaultValue(entity.getDefaultValue());
        dto.setRequired(entity.getRequired());
        dto.setIsUnique(entity.getIsUnique());
        dto.setValidationRule(entity.getValidationRule());
        dto.setListVisible(entity.getListVisible());
        dto.setSearchable(entity.getSearchable());
        dto.setSortOrder(entity.getSortOrder());
        dto.setFieldGroup(entity.getFieldGroup());
        dto.setReferenceType(entity.getReferenceType());
        dto.setReferenceEntity(entity.getReferenceEntity());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public String getFieldCode() { return fieldCode; }
    public void setFieldCode(String fieldCode) { this.fieldCode = fieldCode; }
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    public String getFieldType() { return fieldType; }
    public void setFieldType(String fieldType) { this.fieldType = fieldType; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public Boolean getRequired() { return required; }
    public void setRequired(Boolean required) { this.required = required; }
    public Boolean getIsUnique() { return isUnique; }
    public void setIsUnique(Boolean isUnique) { this.isUnique = isUnique; }
    public String getValidationRule() { return validationRule; }
    public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
    public Boolean getListVisible() { return listVisible; }
    public void setListVisible(Boolean listVisible) { this.listVisible = listVisible; }
    public Boolean getSearchable() { return searchable; }
    public void setSearchable(Boolean searchable) { this.searchable = searchable; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getFieldGroup() { return fieldGroup; }
    public void setFieldGroup(String fieldGroup) { this.fieldGroup = fieldGroup; }
    public String getReferenceType() { return referenceType; }
    public void setReferenceType(String referenceType) { this.referenceType = referenceType; }
    public String getReferenceEntity() { return referenceEntity; }
    public void setReferenceEntity(String referenceEntity) { this.referenceEntity = referenceEntity; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}