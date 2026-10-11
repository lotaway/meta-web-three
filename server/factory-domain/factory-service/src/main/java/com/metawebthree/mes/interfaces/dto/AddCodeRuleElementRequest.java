package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;

public class AddCodeRuleElementRequest {
    
    @NotBlank(message = "元素类型不能为空")
    private String elementType;
    
    @NotBlank(message = "元素值不能为空")
    private String elementValue;
    
    private String fieldName;
    
    public String getElementType() { return elementType; }
    public void setElementType(String elementType) { this.elementType = elementType; }
    public String getElementValue() { return elementValue; }
    public void setElementValue(String elementValue) { this.elementValue = elementValue; }
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
}