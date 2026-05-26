package com.metawebthree.mes.interfaces.dto;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class ProcessParameterDTO {
    
    private Long id;
    private String paramCode;
    private String paramName;
    private Long routeId;
    private String routeCode;
    private Integer stepNo;
    private String stepCode;
    private String paramType;
    private String dataType;
    private String unit;
    private BigDecimal standardValue;
    private BigDecimal upperLimit;
    private BigDecimal lowerLimit;
    private String collectionMethod;
    private String deviceAddress;
    private Boolean isRequired;
    private String validationRule;
    private BigDecimal alarmThreshold;
    private Integer displayOrder;
    private String paramGroup;
    private String remark;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static ProcessParameterDTO fromEntity(
            com.metawebthree.mes.domain.entity.ProcessParameter entity) {
        if (entity == null) return null;
        
        ProcessParameterDTO dto = new ProcessParameterDTO();
        dto.setId(entity.getId());
        dto.setParamCode(entity.getParamCode());
        dto.setParamName(entity.getParamName());
        dto.setRouteId(entity.getRouteId());
        dto.setRouteCode(entity.getRouteCode());
        dto.setStepNo(entity.getStepNo());
        dto.setStepCode(entity.getStepCode());
        dto.setParamType(entity.getParamType() != null ? entity.getParamType().name() : null);
        dto.setDataType(entity.getDataType() != null ? entity.getDataType().name() : null);
        dto.setUnit(entity.getUnit());
        dto.setStandardValue(entity.getStandardValue());
        dto.setUpperLimit(entity.getUpperLimit());
        dto.setLowerLimit(entity.getLowerLimit());
        dto.setCollectionMethod(entity.getCollectionMethod() != null ? entity.getCollectionMethod().name() : null);
        dto.setDeviceAddress(entity.getDeviceAddress());
        dto.setIsRequired(entity.getIsRequired());
        dto.setValidationRule(entity.getValidationRule());
        dto.setAlarmThreshold(entity.getAlarmThreshold());
        dto.setDisplayOrder(entity.getDisplayOrder());
        dto.setParamGroup(entity.getParamGroup());
        dto.setRemark(entity.getRemark());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getParamCode() { return paramCode; }
    public void setParamCode(String paramCode) { this.paramCode = paramCode; }
    public String getParamName() { return paramName; }
    public void setParamName(String paramName) { this.paramName = paramName; }
    public Long getRouteId() { return routeId; }
    public void setRouteId(Long routeId) { this.routeId = routeId; }
    public String getRouteCode() { return routeCode; }
    public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    public Integer getStepNo() { return stepNo; }
    public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
    public String getStepCode() { return stepCode; }
    public void setStepCode(String stepCode) { this.stepCode = stepCode; }
    public String getParamType() { return paramType; }
    public void setParamType(String paramType) { this.paramType = paramType; }
    public String getDataType() { return dataType; }
    public void setDataType(String dataType) { this.dataType = dataType; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public BigDecimal getStandardValue() { return standardValue; }
    public void setStandardValue(BigDecimal standardValue) { this.standardValue = standardValue; }
    public BigDecimal getUpperLimit() { return upperLimit; }
    public void setUpperLimit(BigDecimal upperLimit) { this.upperLimit = upperLimit; }
    public BigDecimal getLowerLimit() { return lowerLimit; }
    public void setLowerLimit(BigDecimal lowerLimit) { this.lowerLimit = lowerLimit; }
    public String getCollectionMethod() { return collectionMethod; }
    public void setCollectionMethod(String collectionMethod) { this.collectionMethod = collectionMethod; }
    public String getDeviceAddress() { return deviceAddress; }
    public void setDeviceAddress(String deviceAddress) { this.deviceAddress = deviceAddress; }
    public Boolean getIsRequired() { return isRequired; }
    public void setIsRequired(Boolean isRequired) { this.isRequired = isRequired; }
    public String getValidationRule() { return validationRule; }
    public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
    public BigDecimal getAlarmThreshold() { return alarmThreshold; }
    public void setAlarmThreshold(BigDecimal alarmThreshold) { this.alarmThreshold = alarmThreshold; }
    public Integer getDisplayOrder() { return displayOrder; }
    public void setDisplayOrder(Integer displayOrder) { this.displayOrder = displayOrder; }
    public String getParamGroup() { return paramGroup; }
    public void setParamGroup(String paramGroup) { this.paramGroup = paramGroup; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}