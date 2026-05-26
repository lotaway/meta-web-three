package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;

public class CreateDataDictionaryRequest {
    
    @NotBlank(message = "字典编码不能为空")
    private String dictCode;
    
    @NotBlank(message = "字典名称不能为空")
    private String dictName;
    
    private String description;
    
    public String getDictCode() { return dictCode; }
    public void setDictCode(String dictCode) { this.dictCode = dictCode; }
    public String getDictName() { return dictName; }
    public void setDictName(String dictName) { this.dictName = dictName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}