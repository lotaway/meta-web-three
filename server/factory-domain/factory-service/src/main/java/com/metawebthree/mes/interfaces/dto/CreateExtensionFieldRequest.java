package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;

public class CreateExtensionFieldRequest {
    
    @NotBlank(message = "实体类型不能为空")
    private String entityType;
    
    @NotBlank(message = "字段编码不能为空")
    private String fieldCode;
    
    @NotBlank(message = "字段名称不能为空")
    private String fieldName;
    
    @NotBlank(message = "字段类型不能为空")
    private String fieldType;
    
    private String fieldGroup;
    private Boolean required;
    private String defaultValue;
    private String validationRule;
    
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public String getFieldCode() { return fieldCode; }
    public void setFieldCode(String fieldCode) { this.fieldCode = fieldCode; }
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    public String getFieldType() { return fieldType; }
    public void setFieldType(String fieldType) { this.fieldType = fieldType; }
    public String getFieldGroup() { return fieldGroup; }
    public void setFieldGroup(String fieldGroup) { this.fieldGroup = fieldGroup; }
    public Boolean getRequired() { return required; }
    public void setRequired(Boolean required) { this.required = required; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public String getValidationRule() { return validationRule; }
    public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
}