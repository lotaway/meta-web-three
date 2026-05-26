package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.Map;

public class SetExtensionFieldValuesRequest {
    
    @NotEmpty(message = "字段值不能为空")
    private Map<String, String> fieldValues;
    
    public Map<String, String> getFieldValues() { return fieldValues; }
    public void setFieldValues(Map<String, String> fieldValues) { this.fieldValues = fieldValues; }
}