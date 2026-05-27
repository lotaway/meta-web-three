package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class AndonType {
    
    private Long id;
    private String typeCode;
    private String typeName;
    private AndonCategory category;
    private String description;
    private Boolean requirePhoto;
    private Boolean requireConfirm;
    private Integer defaultEscalationMinutes;
    private String defaultProcessTemplate;
    private AndonStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum AndonCategory {
        EQUIPMENT,
        MATERIAL,
        QUALITY,
        PERSONNEL,
        SAFETY,
        OTHER
    }
    
    public enum AndonStatus {
        ACTIVE, INACTIVE
    }
    
    public static AndonType create(String typeCode, String typeName, AndonCategory category) {
        AndonType type = new AndonType();
        type.typeCode = typeCode;
        type.typeName = typeName;
        type.category = category;
        type.status = AndonStatus.ACTIVE;
        type.requirePhoto = false;
        type.requireConfirm = false;
        type.defaultEscalationMinutes = 30;
        type.sortOrder = 0;
        type.createdAt = LocalDateTime.now();
        type.updatedAt = LocalDateTime.now();
        return type;
    }
    
    public void update(String typeName, AndonCategory category, String description,
                       Boolean requirePhoto, Boolean requireConfirm, 
                       Integer defaultEscalationMinutes, String defaultProcessTemplate) {
        this.typeName = typeName;
        this.category = category;
        this.description = description;
        this.requirePhoto = requirePhoto;
        this.requireConfirm = requireConfirm;
        this.defaultEscalationMinutes = defaultEscalationMinutes;
        this.defaultProcessTemplate = defaultProcessTemplate;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = AndonStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = AndonStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isActive() {
        return this.status == AndonStatus.ACTIVE;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public AndonCategory getCategory() { return category; }
    public void setCategory(AndonCategory category) { this.category = category; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Boolean getRequirePhoto() { return requirePhoto; }
    public void setRequirePhoto(Boolean requirePhoto) { this.requirePhoto = requirePhoto; }
    public Boolean getRequireConfirm() { return requireConfirm; }
    public void setRequireConfirm(Boolean requireConfirm) { this.requireConfirm = requireConfirm; }
    public Integer getDefaultEscalationMinutes() { return defaultEscalationMinutes; }
    public void setDefaultEscalationMinutes(Integer defaultEscalationMinutes) { this.defaultEscalationMinutes = defaultEscalationMinutes; }
    public String getDefaultProcessTemplate() { return defaultProcessTemplate; }
    public void setDefaultProcessTemplate(String defaultProcessTemplate) { this.defaultProcessTemplate = defaultProcessTemplate; }
    public AndonStatus getStatus() { return status; }
    public void setStatus(AndonStatus status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}