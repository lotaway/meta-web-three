package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.QcConstants;

import java.time.LocalDateTime;

public class QcInspectionItem {
    
    private Long id;
    private String itemCode;
    private String itemName;
    private String itemCategory;
    private String dataType;
    private String unit;
    private Double standardValue;
    private Double upperLimit;
    private Double lowerLimit;
    private String inspectionMethod;
    private String inspectionTool;
    private Integer severity;
    private Boolean isMandatory;
    private ItemStatus status;
    private Integer sortOrder;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum DataType {
        NUMERIC, TEXT, BOOLEAN, DATE, SELECT
    }
    
    public enum ItemStatus {
        ACTIVE, INACTIVE
    }
    
    public static QcInspectionItem create(String itemCode, String itemName, String itemCategory) {
        QcInspectionItem item = new QcInspectionItem();
        item.itemCode = itemCode;
        item.itemName = itemName;
        item.itemCategory = itemCategory;
        item.dataType = DataType.NUMERIC.name();
        item.severity = QcConstants.DEFAULT_SEVERITY;
        item.isMandatory = QcConstants.DEFAULT_IS_MANDATORY;
        item.sortOrder = QcConstants.DEFAULT_SORT_ORDER;
        item.createdAt = LocalDateTime.now();
        item.updatedAt = LocalDateTime.now();
        return item;
    }
    
    public void update(String itemName, String itemCategory, String dataType, String unit,
            Double standardValue, Double upperLimit, Double lowerLimit, String inspectionMethod,
            String inspectionTool, Integer severity, Boolean isMandatory, String remark) {
        this.itemName = itemName;
        this.itemCategory = itemCategory;
        this.dataType = dataType;
        this.unit = unit;
        this.standardValue = standardValue;
        this.upperLimit = upperLimit;
        this.lowerLimit = lowerLimit;
        this.inspectionMethod = inspectionMethod;
        this.inspectionTool = inspectionTool;
        this.severity = severity;
        this.isMandatory = isMandatory;
        this.remark = remark;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = ItemStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = ItemStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isActive() {
        return this.status == ItemStatus.ACTIVE;
    }
    
    public boolean isWithinSpec(Double value) {
        if (value == null) return false;
        if (upperLimit != null && value > upperLimit) return false;
        if (lowerLimit != null && value < lowerLimit) return false;
        return true;
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
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public Double getStandardValue() { return standardValue; }
    public void setStandardValue(Double standardValue) { this.standardValue = standardValue; }
    public Double getUpperLimit() { return upperLimit; }
    public void setUpperLimit(Double upperLimit) { this.upperLimit = upperLimit; }
    public Double getLowerLimit() { return lowerLimit; }
    public void setLowerLimit(Double lowerLimit) { this.lowerLimit = lowerLimit; }
    public String getInspectionMethod() { return inspectionMethod; }
    public void setInspectionMethod(String inspectionMethod) { this.inspectionMethod = inspectionMethod; }
    public String getInspectionTool() { return inspectionTool; }
    public void setInspectionTool(String inspectionTool) { this.inspectionTool = inspectionTool; }
    public Integer getSeverity() { return severity; }
    public void setSeverity(Integer severity) { this.severity = severity; }
    public Boolean getIsMandatory() { return isMandatory; }
    public void setIsMandatory(Boolean isMandatory) { this.isMandatory = isMandatory; }
    public ItemStatus getStatus() { return status; }
    public void setStatus(ItemStatus status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}