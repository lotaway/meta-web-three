package com.metawebthree.mes.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 工艺参数配置
 * 用于定义生产工艺过程中的参数标准
 * 对应 SPEC 文档 4.2 工艺参数配置
 */
public class ProcessParameter {
    
    private Long id;
    
    /**
     * 参数编码（唯一标识）
     */
    private String paramCode;
    
    /**
     * 参数名称
     */
    private String paramName;
    
    /**
     * 所属工艺路线ID
     */
    private Long routeId;
    
    /**
     * 工艺路线编码
     */
    private String routeCode;
    
    /**
     * 工序序号
     */
    private Integer stepNo;
    
    /**
     * 工序编码
     */
    private String stepCode;
    
    /**
     * 参数类型
     */
    private ParamType paramType;
    
    /**
     * 数据类型
     */
    private DataType dataType;
    
    /**
     * 单位
     */
    private String unit;
    
    /**
     * 标准值
     */
    private BigDecimal standardValue;
    
    /**
     * 规格上限
     */
    private BigDecimal upperLimit;
    
    /**
     * 规格下限
     */
    private BigDecimal lowerLimit;
    
    /**
     * 采集方式: MANUAL, AUTO_SENSOR, PLC, barcode
     */
    private CollectionMethod collectionMethod;
    
    /**
     * 采集设备地址
     */
    private String deviceAddress;
    
    /**
     * 是否必填
     */
    private Boolean required;
    
    /**
     * 校验规则（正则表达式）
     */
    private String validationRule;
    
    /**
     * 超差报警阈值（百分比）
     */
    private BigDecimal alarmThreshold;
    
    /**
     * 显示顺序
     */
    private Integer displayOrder;
    
    /**
     * 参数分组
     */
    private String paramGroup;
    
    /**
     * 备注说明
     */
    private String remark;
    
    /**
     * 状态: ACTIVE, INACTIVE
     */
    private ParamStatus status;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
    
    /**
     * 参数类型枚举
     */
    public enum ParamType {
        TEMPERATURE,       // 温度
        PRESSURE,          // 压力
        SPEED,             // 速度
        TIME,              // 时间
        CURRENT,           // 电流
        VOLTAGE,           // 电压
        FORCE,             // 力
        LENGTH,            // 长度
        ANGLE,             // 角度
        WEIGHT,            // 重量
        VOLUME,            // 体积
        SPEED_PER_MINUTE,  // 转速
        HUMIDITY,          // 湿度
        QUALITY,           // 质量参数
        COUNT,             // 计数
        OTHER              // 其他
    }
    
    /**
     * 数据类型枚举
     */
    public enum DataType {
        INTEGER,   // 整数
        DECIMAL,   // 小数
        TEXT,      // 文本
        BOOLEAN    // 布尔
    }
    
    /**
     * 采集方式枚举
     */
    public enum CollectionMethod {
        MANUAL,      // 人工录入
        AUTO_SENSOR, // 传感器自动采集
        PLC,         // PLC采集
        BARCODE      // 条码扫描
    }
    
    /**
     * 参数状态枚举
     */
    public enum ParamStatus {
        ACTIVE, INACTIVE
    }
    
    /**
     * 创建工艺参数
     */
    public void create(String paramCode, String paramName, Long routeId, 
                       String routeCode, Integer stepNo, String stepCode,
                       ParamType paramType, DataType dataType) {
        this.paramCode = paramCode;
        this.paramName = paramName;
        this.routeId = routeId;
        this.routeCode = routeCode;
        this.stepNo = stepNo;
        this.stepCode = stepCode;
        this.paramType = paramType;
        this.dataType = dataType;
        this.required = false;
        this.status = ParamStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 校验参数值是否在规格范围内
     */
    public boolean validateValue(BigDecimal value) {
        if (value == null) {
            return !required;
        }
        
        if (upperLimit != null && value.compareTo(upperLimit) > 0) {
            return false;
        }
        
        if (lowerLimit != null && value.compareTo(lowerLimit) < 0) {
            return false;
        }
        
        return true;
    }
    
    /**
     * 计算偏差百分比
     */
    public BigDecimal calculateDeviation(BigDecimal actualValue) {
        if (standardValue == null || actualValue == null || standardValue.compareTo(BigDecimal.ZERO) == 0) {
            return null;
        }
        return actualValue.subtract(standardValue)
                          .divide(standardValue, 4, BigDecimal.ROUND_HALF_UP)
                          .multiply(new BigDecimal("100"));
    }
    
    /**
     * 判断是否超差
     */
    public boolean isOutOfTolerance(BigDecimal actualValue) {
        BigDecimal deviation = calculateDeviation(actualValue);
        if (deviation == null || alarmThreshold == null) {
            return false;
        }
        return deviation.abs().compareTo(alarmThreshold) > 0;
    }
    
    // Getters and Setters
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
    public Boolean getRequired() { return required; }
    public void setRequired(Boolean required) { this.required = required; }
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