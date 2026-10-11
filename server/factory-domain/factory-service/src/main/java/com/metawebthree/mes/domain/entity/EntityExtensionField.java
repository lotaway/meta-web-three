package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class EntityExtensionField {
    
    private Long id;
    private String entityType;
    private String fieldCode;
    private String fieldName;
    private FieldType fieldType;
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
    private FieldStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum FieldType {
        TEXT, TEXTAREA, NUMBER, DATE, DATETIME, SELECT, MULTI_SELECT, CHECKBOX, SWITCH, REFERENCE
    }
    
    public enum FieldStatus {
        ACTIVE, INACTIVE
    }
    
    public static EntityExtensionField create(String entityType, String fieldCode, String fieldName,
                                               FieldType fieldType, String fieldGroup) {
        EntityExtensionField field = new EntityExtensionField();
        field.entityType = entityType;
        field.fieldCode = fieldCode;
        field.fieldName = fieldName;
        field.fieldType = fieldType;
        field.fieldGroup = fieldGroup;
        field.required = false;
        field.isUnique = false;
        field.listVisible = true;
        field.searchable = false;
        field.status = FieldStatus.ACTIVE;
        field.createdAt = LocalDateTime.now();
        field.updatedAt = LocalDateTime.now();
        return field;
    }
    
    public boolean validateValue(String value) {
        if (required && (value == null || value.isEmpty())) {
            return false;
        }
        if (validationRule != null && !validationRule.isEmpty() && value != null && !value.isEmpty()) {
            return value.matches(validationRule);
        }
        return true;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public String getFieldCode() { return fieldCode; }
    public void setFieldCode(String fieldCode) { this.fieldCode = fieldCode; }
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    public FieldType getFieldType() { return fieldType; }
    public void setFieldType(FieldType fieldType) { this.fieldType = fieldType; }
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
    public FieldStatus getStatus() { return status; }
    public void setStatus(FieldStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}