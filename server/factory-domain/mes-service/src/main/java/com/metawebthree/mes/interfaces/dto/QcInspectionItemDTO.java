package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.QcInspectionItem;
import java.time.LocalDateTime;

public class QcInspectionItemDTO {
    
    private Long id;
    private String itemCode;
    private String itemName;
    private String itemCategory;
    private String itemCategoryName;
    private String dataType;
    private String dataTypeName;
    private String unit;
    private Double standardValue;
    private Double upperLimit;
    private Double lowerLimit;
    private String inspectionMethod;
    private String inspectionTool;
    private Integer severity;
    private String severityName;
    private Boolean isMandatory;
    private String status;
    private String statusName;
    private Integer sortOrder;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static QcInspectionItemDTO fromEntity(QcInspectionItem entity) {
        if (entity == null) return null;
        
        QcInspectionItemDTO dto = new QcInspectionItemDTO();
        dto.setId(entity.getId());
        dto.setItemCode(entity.getItemCode());
        dto.setItemName(entity.getItemName());
        dto.setItemCategory(entity.getItemCategory());
        dto.setItemCategoryName(getItemCategoryName(entity.getItemCategory()));
        dto.setDataType(entity.getDataType());
        dto.setDataTypeName(getDataTypeName(entity.getDataType()));
        dto.setUnit(entity.getUnit());
        dto.setStandardValue(entity.getStandardValue());
        dto.setUpperLimit(entity.getUpperLimit());
        dto.setLowerLimit(entity.getLowerLimit());
        dto.setInspectionMethod(entity.getInspectionMethod());
        dto.setInspectionTool(entity.getInspectionTool());
        dto.setSeverity(entity.getSeverity());
        dto.setSeverityName(getSeverityName(entity.getSeverity()));
        dto.setIsMandatory(entity.getIsMandatory());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setStatusName(getStatusName(entity.getStatus()));
        dto.setSortOrder(entity.getSortOrder());
        dto.setRemark(entity.getRemark());
        return dto;
    }
    
    public QcInspectionItem toEntity() {
        QcInspectionItem entity = new QcInspectionItem();
        entity.setId(this.id);
        entity.setItemCode(this.itemCode);
        entity.setItemName(this.itemName);
        entity.setItemCategory(this.itemCategory);
        entity.setDataType(this.dataType);
        entity.setUnit(this.unit);
        entity.setStandardValue(this.standardValue);
        entity.setUpperLimit(this.upperLimit);
        entity.setLowerLimit(this.lowerLimit);
        entity.setInspectionMethod(this.inspectionMethod);
        entity.setInspectionTool(this.inspectionTool);
        entity.setSeverity(this.severity);
        entity.setIsMandatory(this.isMandatory);
        entity.setStatus(this.status != null ? Enum.valueOf(QcInspectionItem.ItemStatus.class, this.status) : null);
        entity.setSortOrder(this.sortOrder);
        entity.setRemark(this.remark);
        return entity;
    }
    
    private static String getItemCategoryName(String category) {
        if (category == null) return null;
        switch (category) {
            case "appearance": return "外观检验";
            case "dimension": return "尺寸检验";
            case "function": return "功能检验";
            case "performance": return "性能检验";
            case "safety": return "安全检验";
            default: return category;
        }
    }
    
    private static String getDataTypeName(String dataType) {
        if (dataType == null) return null;
        switch (dataType) {
            case "NUMERIC": return "数值";
            case "TEXT": return "文本";
            case "BOOLEAN": return "布尔";
            case "DATE": return "日期";
            case "SELECT": return "选择";
            default: return dataType;
        }
    }
    
    private static String getSeverityName(Integer severity) {
        if (severity == null) return null;
        switch (severity) {
            case 1: return "轻微";
            case 2: return "一般";
            case 3: return "严重";
            case 4: return "致命";
            default: return severity.toString();
        }
    }
    
    private static String getStatusName(QcInspectionItem.ItemStatus status) {
        if (status == null) return null;
        switch (status) {
            case ACTIVE: return "启用";
            case INACTIVE: return "停用";
            default: return status.name();
        }
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getItemCode() { return itemCode; }
    public void setItemCode(String itemCode) { this.itemCode = itemCode; }
    public String getItemName() { return itemName; }
    public void setItemName(String itemName) { this.itemName = itemName; }
    public String getItemCategory() { return itemCategory; }
    public void setItemCategory(String itemCategory) { this.itemCategory = itemCategory; }
    public String getItemCategoryName() { return itemCategoryName; }
    public void setItemCategoryName(String itemCategoryName) { this.itemCategoryName = itemCategoryName; }
    public String getDataType() { return dataType; }
    public void setDataType(String dataType) { this.dataType = dataType; }
    public String getDataTypeName() { return dataTypeName; }
    public void setDataTypeName(String dataTypeName) { this.dataTypeName = dataTypeName; }
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
    public String getSeverityName() { return severityName; }
    public void setSeverityName(String severityName) { this.severityName = severityName; }
    public Boolean getIsMandatory() { return isMandatory; }
    public void setIsMandatory(Boolean isMandatory) { this.isMandatory = isMandatory; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getStatusName() { return statusName; }
    public void setStatusName(String statusName) { this.statusName = statusName; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}