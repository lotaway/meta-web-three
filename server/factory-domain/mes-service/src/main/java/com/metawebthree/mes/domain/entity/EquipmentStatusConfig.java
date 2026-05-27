package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class EquipmentStatusConfig {
    private Long id;
    private Long equipmentTypeId;
    private String statusCode;
    private String statusName;
    private String statusCategory;
    private Boolean isInitial;
    private Boolean isFinal;
    private Boolean isAlarm;
    private String color;
    private String icon;
    private String description;
    private Integer sortOrder;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long equipmentTypeId, String statusCode, String statusName, String statusCategory) {
        this.equipmentTypeId = equipmentTypeId;
        this.statusCode = statusCode;
        this.statusName = statusName;
        this.statusCategory = statusCategory;
        this.isInitial = false;
        this.isFinal = false;
        this.isAlarm = false;
        this.isActive = true;
        this.sortOrder = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setAsInitial() {
        this.isInitial = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void setAsFinal() {
        this.isFinal = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void setAsAlarm() {
        this.isAlarm = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isAvailableForTransition() {
        return isActive && !isFinal;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getEquipmentTypeId() { return equipmentTypeId; }
    public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
    public String getStatusCode() { return statusCode; }
    public void setStatusCode(String statusCode) { this.statusCode = statusCode; }
    public String getStatusName() { return statusName; }
    public void setStatusName(String statusName) { this.statusName = statusName; }
    public String getStatusCategory() { return statusCategory; }
    public void setStatusCategory(String statusCategory) { this.statusCategory = statusCategory; }
    public Boolean getIsInitial() { return isInitial; }
    public void setIsInitial(Boolean isInitial) { this.isInitial = isInitial; }
    public Boolean getIsFinal() { return isFinal; }
    public void setIsFinal(Boolean isFinal) { this.isFinal = isFinal; }
    public Boolean getIsAlarm() { return isAlarm; }
    public void setIsAlarm(Boolean isAlarm) { this.isAlarm = isAlarm; }
    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }
    public String getIcon() { return icon; }
    public void setIcon(String icon) { this.icon = icon; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}