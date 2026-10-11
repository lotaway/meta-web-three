package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;

public class EntityExtensionFieldValueDTO {
    
    private Long id;
    private String entityType;
    private Long entityId;
    private String fieldCode;
    private String fieldValue;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static EntityExtensionFieldValueDTO fromEntity(
            com.metawebthree.mes.domain.entity.EntityExtensionFieldValue entity) {
        if (entity == null) return null;
        
        EntityExtensionFieldValueDTO dto = new EntityExtensionFieldValueDTO();
        dto.setId(entity.getId());
        dto.setEntityType(entity.getEntityType());
        dto.setEntityId(entity.getEntityId());
        dto.setFieldCode(entity.getFieldCode());
        dto.setFieldValue(entity.getFieldValue());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
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
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}