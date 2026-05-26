package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

/**
 * 实体扩展字段值
 * 存储每个实体实例的扩展字段值
 */
public class EntityExtensionFieldValue {
    
    private Long id;
    
    /**
     * 实体类型
     */
    private String entityType;
    
    /**
     * 实体ID
     */
    private Long entityId;
    
    /**
     * 字段编码
     */
    private String fieldCode;
    
    /**
     * 字段值
     */
    private String fieldValue;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
    
    /**
     * 创建扩展字段值
     */
    public void create(String entityType, Long entityId, String fieldCode, String fieldValue) {
        this.entityType = entityType;
        this.entityId = entityId;
        this.fieldCode = fieldCode;
        this.fieldValue = fieldValue;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 更新字段值
     */
    public void updateValue(String newValue) {
        this.fieldValue = newValue;
        this.updatedAt = LocalDateTime.now();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public Long getEntityId() { return entityId; }
    public void setEntityId(Long entityId) { this.entityId = entityId; }
    public String getFieldCode() { return fieldCode; }
    public void setFieldCode(String fieldCode) { this.fieldCode = fieldCode; }
    public String getFieldValue() { return fieldValue; }
    public void setFieldValue(String fieldValue) { this.fieldValue = fieldValue; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}