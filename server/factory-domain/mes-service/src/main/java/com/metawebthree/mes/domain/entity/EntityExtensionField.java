package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

/**
 * 实体扩展字段定义
 * 用于实现配置化MES的扩展字段机制
 */
public class EntityExtensionField {
    
    private Long id;
    
    /**
     * 所属实体类型: work_order, product, material, equipment, qc_inspection
     */
    private String entityType;
    
    /**
     * 字段编码（唯一标识）
     */
    private String fieldCode;
    
    /**
     * 字段名称
     */
    private String fieldName;
    
    /**
     * 字段类型: TEXT, NUMBER, DATE, DATETIME, SELECT, MULTI_SELECT, CHECKBOX, SWITCH, REFERENCE
     */
    private FieldType fieldType;
    
    /**
     * 默认值
     */
    private String defaultValue;
    
    /**
     * 是否必填
     */
    private Boolean required;
    
    /**
     * 是否唯一
     */
    private Boolean unique;
    
    /**
     * 校验规则（正则表达式）
     */
    private String validationRule;
    
    /**
     * 是否在列表中显示
     */
    private Boolean listVisible;
    
    /**
     * 是否可搜索
     */
    private Boolean searchable;
    
    /**
     * 排序序号
     */
    private Integer sortOrder;
    
    /**
     * 字段分组
     */
    private String fieldGroup;
    
    /**
     * 引用类型（当fieldType为REFERENCE时）
     */
    private String referenceType;
    
    /**
     * 引用实体（当fieldType为REFERENCE时）
     */
    private String referenceEntity;
    
    /**
     * 状态: ACTIVE, INACTIVE
     */
    private FieldStatus status;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
    
    public enum FieldType {
        TEXT,           // 单行文本
        TEXTAREA,       // 多行文本
        NUMBER,         // 数字
        DATE,           // 日期
        DATETIME,       // 日期时间
        SELECT,         // 下拉单选
        MULTI_SELECT,   // 下拉多选
        CHECKBOX,       // 复选框
        SWITCH,         // 开关
        REFERENCE       // 关联引用
    }
    
    public enum FieldStatus {
        ACTIVE, INACTIVE
    }
    
    /**
     * 创建扩展字段定义
     */
    public void create(String entityType, String fieldCode, String fieldName, 
                       FieldType fieldType, String fieldGroup) {
        this.entityType = entityType;
        this.fieldCode = fieldCode;
        this.fieldName = fieldName;
        this.fieldType = fieldType;
        this.fieldGroup = fieldGroup;
        this.required = false;
        this.unique = false;
        this.listVisible = true;
        this.searchable = false;
        this.status = FieldStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 校验字段值
     */
    public boolean validateValue(String value) {
        if (required && (value == null || value.isEmpty())) {
            return false;
        }
        if (validationRule != null && !validationRule.isEmpty() && value != null && !value.isEmpty()) {
            return value.matches(validationRule);
        }
        return true;
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public String getFieldCode() { return fieldCode; }
    public void setFieldCode(String fieldCode) { this.fieldCode = fieldCode; }
    public String getFieldName() { return fieldName; }
    public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    public FieldType getFieldType() { return fieldType; }
    public void setFieldType(FieldType fieldType) { this.fieldType = fieldType; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public Boolean getRequired() { return required; }
    public void setRequired(Boolean required) { this.required = required; }
    public Boolean getUnique() { return unique; }
    public void setUnique(Boolean unique) { this.unique = unique; }
    public String getValidationRule() { return validationRule; }
    public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
    public Boolean getListVisible() { return listVisible; }
    public void setListVisible(Boolean listVisible) { this.listVisible = listVisible; }
    public Boolean getSearchable() { return searchable; }
    public void setSearchable(Boolean searchable) { this.searchable = searchable; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getFieldGroup() { return fieldGroup; }
    public void setFieldGroup(String fieldGroup) { this.fieldGroup = fieldGroup; }
    public String getReferenceType() { return referenceType; }
    public void setReferenceType(String referenceType) { this.referenceType = referenceType; }
    public String getReferenceEntity() { return referenceEntity; }
    public void setReferenceEntity(String referenceEntity) { this.referenceEntity = referenceEntity; }
    public FieldStatus getStatus() { return status; }
    public void setStatus(FieldStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}