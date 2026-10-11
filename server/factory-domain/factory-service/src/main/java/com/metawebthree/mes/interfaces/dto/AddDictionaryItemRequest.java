package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;

public class AddDictionaryItemRequest {
    
    @NotBlank(message = "项编码不能为空")
    private String itemCode;
    
    @NotBlank(message = "项标签不能为空")
    private String itemLabel;
    
    private String parentItemCode;
    private Integer sortOrder;
    
    public String getItemCode() { return itemCode; }
    public void setItemCode(String itemCode) { this.itemCode = itemCode; }
    public String getItemLabel() { return itemLabel; }
    public void setItemLabel(String itemLabel) { this.itemLabel = itemLabel; }
    public String getParentItemCode() { return parentItemCode; }
    public void setParentItemCode(String parentItemCode) { this.parentItemCode = parentItemCode; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
}