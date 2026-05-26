package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class EntityExtensionFieldValue {

    private Long id;
    private String entityType;
    private Long entityId;
    private String fieldCode;
    private String fieldValue;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static EntityExtensionFieldValue create(String entityType, Long entityId, String fieldCode, String fieldValue) {
        EntityExtensionFieldValue value = new EntityExtensionFieldValue();
        value.entityType = entityType;
        value.entityId = entityId;
        value.fieldCode = fieldCode;
        value.fieldValue = fieldValue;
        value.createdAt = LocalDateTime.now();
        value.updatedAt = LocalDateTime.now();
        return value;
    }

    public void updateValue(String newValue) {
        this.fieldValue = newValue;
        this.updatedAt = LocalDateTime.now();
    }

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