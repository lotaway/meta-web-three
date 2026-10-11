package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class ChecklistItem {
    
    private Long id;
    private String itemCode;
    private String itemName;
    private String itemCategory;
    private String dataType;
    private String standardValue;
    private String upperLimit;
    private String lowerLimit;
    private String unit;
    private String checkMethod;
    private String abnormalJudgment;
    private Boolean isMandatory;
    private Integer sortOrder;
    private ItemStatus status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum ItemStatus {
        ACTIVE, INACTIVE
    }
    
    public enum DataType {
        TEXT, NUMBER, BOOLEAN, DATE, SELECT, MULTI_SELECT
    }
    
    public static ChecklistItem create(String itemCode, String itemName, String itemCategory) {
        ChecklistItem item = new ChecklistItem();
        item.itemCode = itemCode;
        item.itemName = itemName;
        item.itemCategory = itemCategory;
        item.dataType = DataType.TEXT.name();
        item.isMandatory = true;
        item.status = ItemStatus.ACTIVE;
        item.sortOrder = 0;
        item.createdAt = LocalDateTime.now();
        item.updatedAt = LocalDateTime.now();
        return item;
    }
    
    public void update(String itemName, String itemCategory, String dataType,
            String standardValue, String upperLimit, String lowerLimit, String unit,
            String checkMethod, String abnormalJudgment, Boolean isMandatory, Integer sortOrder) {
        this.itemName = itemName;
        this.itemCategory = itemCategory;
        this.dataType = dataType;
        this.standardValue = standardValue;
        this.upperLimit = upperLimit;
        this.lowerLimit = lowerLimit;
        this.unit = unit;
        this.checkMethod = checkMethod;
        this.abnormalJudgment = abnormalJudgment;
        this.isMandatory = isMandatory;
        this.sortOrder = sortOrder;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = ItemStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = ItemStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isAbnormal(Object value) {
        if (value == null || this.dataType == null) {
            return false;
        }
        
        if (DataType.NUMBER.name().equals(this.dataType)) {
            try {
                double numValue = Double.parseDouble(value.toString());
                if (upperLimit != null && numValue > Double.parseDouble(upperLimit)) {
                    return true;
                }
                if (lowerLimit != null && numValue < Double.parseDouble(lowerLimit)) {
                    return true;
                }
            } catch (NumberFormatException e) {
                return false;
            }
        } else if (DataType.BOOLEAN.name().equals(this.dataType)) {
            if (abnormalJudgment != null && !abnormalJudgment.isEmpty()) {
                return !abnormalJudgment.equalsIgnoreCase(value.toString());
            }
        }
        
        return false;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getItemCode() { return itemCode; }
    public void setItemCode(String itemCode) { this.itemCode = itemCode; }
    public String getItemName() { return itemName; }
    public void setItemName(String itemName) { this.itemName = itemName; }
    public String getItemCategory() { return itemCategory; }
    public void setItemCategory(String itemCategory) { this.itemCategory = itemCategory; }
    public String getDataType() { return dataType; }
    public void setDataType(String dataType) { this.dataType = dataType; }
    public String getStandardValue() { return standardValue; }
    public void setStandardValue(String standardValue) { this.standardValue = standardValue; }
    public String getUpperLimit() { return upperLimit; }
    public void setUpperLimit(String upperLimit) { this.upperLimit = upperLimit; }
    public String getLowerLimit() { return lowerLimit; }
    public void setLowerLimit(String lowerLimit) { this.lowerLimit = lowerLimit; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public String getCheckMethod() { return checkMethod; }
    public void setCheckMethod(String checkMethod) { this.checkMethod = checkMethod; }
    public String getAbnormalJudgment() { return abnormalJudgment; }
    public void setAbnormalJudgment(String abnormalJudgment) { this.abnormalJudgment = abnormalJudgment; }
    public Boolean getIsMandatory() { return isMandatory; }
    public void setIsMandatory(Boolean isMandatory) { this.isMandatory = isMandatory; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public ItemStatus getStatus() { return status; }
    public void setStatus(ItemStatus status) { this.status = status; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}