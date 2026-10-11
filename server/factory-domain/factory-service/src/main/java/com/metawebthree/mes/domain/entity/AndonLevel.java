package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class AndonLevel {
    
    private Long id;
    private String levelCode;
    private String levelName;
    private Integer levelValue;
    private Integer responseTimeoutMinutes;
    private Integer handlingTimeoutMinutes;
    private String colorCode;
    private String description;
    private AndonLevelStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum AndonLevelStatus {
        ACTIVE, INACTIVE
    }
    
    public static AndonLevel create(String levelCode, String levelName, Integer levelValue) {
        AndonLevel level = new AndonLevel();
        level.levelCode = levelCode;
        level.levelName = levelName;
        level.levelValue = levelValue;
        level.responseTimeoutMinutes = 30;
        level.handlingTimeoutMinutes = 60;
        level.colorCode = "#FF0000";
        level.status = AndonLevelStatus.ACTIVE;
        level.sortOrder = levelValue;
        level.createdAt = LocalDateTime.now();
        level.updatedAt = LocalDateTime.now();
        return level;
    }
    
    public void update(String levelName, Integer levelValue, Integer responseTimeoutMinutes,
                      Integer handlingTimeoutMinutes, String colorCode, String description) {
        this.levelName = levelName;
        this.levelValue = levelValue;
        this.responseTimeoutMinutes = responseTimeoutMinutes;
        this.handlingTimeoutMinutes = handlingTimeoutMinutes;
        this.colorCode = colorCode;
        this.description = description;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = AndonLevelStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = AndonLevelStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isActive() {
        return this.status == AndonLevelStatus.ACTIVE;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getLevelCode() { return levelCode; }
    public void setLevelCode(String levelCode) { this.levelCode = levelCode; }
    public String getLevelName() { return levelName; }
    public void setLevelName(String levelName) { this.levelName = levelName; }
    public Integer getLevelValue() { return levelValue; }
    public void setLevelValue(Integer levelValue) { this.levelValue = levelValue; }
    public Integer getResponseTimeoutMinutes() { return responseTimeoutMinutes; }
    public void setResponseTimeoutMinutes(Integer responseTimeoutMinutes) { this.responseTimeoutMinutes = responseTimeoutMinutes; }
    public Integer getHandlingTimeoutMinutes() { return handlingTimeoutMinutes; }
    public void setHandlingTimeoutMinutes(Integer handlingTimeoutMinutes) { this.handlingTimeoutMinutes = handlingTimeoutMinutes; }
    public String getColorCode() { return colorCode; }
    public void setColorCode(String colorCode) { this.colorCode = colorCode; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public AndonLevelStatus getStatus() { return status; }
    public void setStatus(AndonLevelStatus status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}