package com.metawebthree.mes.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Optional;

public class ProcessParameter {
    
    private Long id;
    private String paramCode;
    private String paramName;
    private Long routeId;
    private String routeCode;
    private Integer stepNo;
    private String stepCode;
    private ParamType paramType;
    private DataType dataType;
    private String unit;
    private BigDecimal standardValue;
    private BigDecimal upperLimit;
    private BigDecimal lowerLimit;
    private CollectionMethod collectionMethod;
    private String deviceAddress;
    private Boolean isRequired;
    private String validationRule;
    private BigDecimal alarmThreshold;
    private Integer displayOrder;
    private String paramGroup;
    private String remark;
    private ParamStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum ParamType {
        TEMPERATURE,       
        PRESSURE,          
        SPEED,             
        TIME,              
        CURRENT,           
        VOLTAGE,           
        FORCE,             
        LENGTH,            
        ANGLE,             
        WEIGHT,            
        VOLUME,            
        SPEED_PER_MINUTE,  
        HUMIDITY,          
        QUALITY,           
        COUNT,             
        OTHER              
    }
    
    public enum DataType {
        INTEGER,   
        DECIMAL,   
        TEXT,      
        BOOLEAN    
    }
    
    public enum CollectionMethod {
        MANUAL,      
        AUTO_SENSOR, 
        PLC,         
        BARCODE      
    }
    
    public enum ParamStatus {
        ACTIVE, INACTIVE
    }
    
    public static ProcessParameter create(String paramCode, String paramName, Long routeId, 
                                           String routeCode, Integer stepNo, String stepCode,
                                           ParamType paramType, DataType dataType) {
        ProcessParameter param = new ProcessParameter();
        param.paramCode = paramCode;
        param.paramName = paramName;
        param.routeId = routeId;
        param.routeCode = routeCode;
        param.stepNo = stepNo;
        param.stepCode = stepCode;
        param.paramType = paramType;
        param.dataType = dataType;
        param.isRequired = false;
        param.status = ParamStatus.ACTIVE;
        param.createdAt = LocalDateTime.now();
        param.updatedAt = LocalDateTime.now();
        return param;
    }
    
    public static Optional<ProcessParameter> createForQuery(Long id) {
        return Optional.empty();
    }
    
    public boolean validateValue(BigDecimal value) {
        if (value == null) {
            return !isRequired;
        }
        
        if (upperLimit != null && value.compareTo(upperLimit) > 0) {
            return false;
        }
        
        if (lowerLimit != null && value.compareTo(lowerLimit) < 0) {
            return false;
        }
        
        return true;
    }
    
    public BigDecimal calculateDeviation(BigDecimal actualValue) {
        if (standardValue == null || actualValue == null || standardValue.compareTo(BigDecimal.ZERO) == 0) {
            return null;
        }
        return actualValue.subtract(standardValue)
                          .divide(standardValue, 4, BigDecimal.ROUND_HALF_UP)
                          .multiply(new BigDecimal("100"));
    }
    
    public boolean isOutOfTolerance(BigDecimal actualValue) {
        BigDecimal deviation = calculateDeviation(actualValue);
        if (deviation == null || alarmThreshold == null) {
            return false;
        }
        return deviation.abs().compareTo(alarmThreshold) > 0;
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
    public ParamType getParamType() { return paramType; }
    public void setParamType(ParamType paramType) { this.paramType = paramType; }
    public DataType getDataType() { return dataType; }
    public void setDataType(DataType dataType) { this.dataType = dataType; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public BigDecimal getStandardValue() { return standardValue; }
    public void setStandardValue(BigDecimal standardValue) { this.standardValue = standardValue; }
    public BigDecimal getUpperLimit() { return upperLimit; }
    public void setUpperLimit(BigDecimal upperLimit) { this.upperLimit = upperLimit; }
    public BigDecimal getLowerLimit() { return lowerLimit; }
    public void setLowerLimit(BigDecimal lowerLimit) { this.lowerLimit = lowerLimit; }
    public CollectionMethod getCollectionMethod() { return collectionMethod; }
    public void setCollectionMethod(CollectionMethod collectionMethod) { this.collectionMethod = collectionMethod; }
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
    public ParamStatus getStatus() { return status; }
    public void setStatus(ParamStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}