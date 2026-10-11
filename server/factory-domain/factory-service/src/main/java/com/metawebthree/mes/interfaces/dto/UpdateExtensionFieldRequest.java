package com.metawebthree.mes.interfaces.dto;

public class UpdateExtensionFieldRequest {
    
    private String fieldName;
    private String fieldGroup;
    private Boolean required;
    private String defaultValue;
    private String validationRule;
    private Boolean listVisible;
    private Boolean searchable;
    
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    public String getFieldGroup() { return fieldGroup; }
    public void setFieldGroup(String fieldGroup) { this.fieldGroup = fieldGroup; }
    public Boolean getRequired() { return required; }
    public void setRequired(Boolean required) { this.required = required; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public String getValidationRule() { return validationRule; }
    public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
    public Boolean getListVisible() { return listVisible; }
    public void setListVisible(Boolean listVisible) { this.listVisible = listVisible; }
    public Boolean getSearchable() { return searchable; }
    public void setSearchable(Boolean searchable) { this.searchable = searchable; }
}