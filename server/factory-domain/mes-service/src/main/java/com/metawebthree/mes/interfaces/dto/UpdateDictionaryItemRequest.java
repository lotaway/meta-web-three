package com.metawebthree.mes.interfaces.dto;

public class UpdateDictionaryItemRequest {
    
    private String itemLabel;
    private Integer sortOrder;
    
    public String getItemLabel() { return itemLabel; }
    public void setItemLabel(String itemLabel) { this.itemLabel = itemLabel; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
}