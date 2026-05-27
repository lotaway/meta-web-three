package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 设备点检模板
 */
public class EquipmentChecklistTemplate {
    
    private Long id;
    private String templateCode;
    private String templateName;
    private String equipmentTypeCode;
    private String checkPeriodType;
    private Integer checkPeriodValue;
    private String checkPeriodUnit;
    private Long runningHoursThreshold;
    private String alertBeforeHours;
    private TemplateStatus status;
    private Integer version;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<ChecklistItem> items = new ArrayList<>();
    
    public enum TemplateStatus {
        DRAFT, EFFECTIVE, CANCELLED
    }
    
    public enum CheckPeriodType {
        DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY, RUNNING_HOURS
    }
    
    public static EquipmentChecklistTemplate create(String templateCode, String templateName, 
            String equipmentTypeCode, String checkPeriodType) {
        EquipmentChecklistTemplate template = new EquipmentChecklistTemplate();
        template.templateCode = templateCode;
        template.templateName = templateName;
        template.equipmentTypeCode = equipmentTypeCode;
        template.checkPeriodType = checkPeriodType;
        template.checkPeriodValue = 1;
        template.checkPeriodUnit = "DAY";
        template.status = TemplateStatus.DRAFT;
        template.version = 1;
        template.createdAt = LocalDateTime.now();
        template.updatedAt = LocalDateTime.now();
        return template;
    }
    
    public void update(String templateName, String equipmentTypeCode, String checkPeriodType,
            Integer checkPeriodValue, String checkPeriodUnit, Long runningHoursThreshold,
            String alertBeforeHours, String remark) {
        this.templateName = templateName;
        this.equipmentTypeCode = equipmentTypeCode;
        this.checkPeriodType = checkPeriodType;
        this.checkPeriodValue = checkPeriodValue;
        this.checkPeriodUnit = checkPeriodUnit;
        this.runningHoursThreshold = runningHoursThreshold;
        this.alertBeforeHours = alertBeforeHours;
        this.remark = remark;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = TemplateStatus.EFFECTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void cancel() {
        this.status = TemplateStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addItem(ChecklistItem item) {
        this.items.add(item);
    }
    
    public void removeItem(Long itemId) {
        this.items.removeIf(item -> item.getId().equals(itemId));
    }
    
    public boolean isEffective() {
        return this.status == TemplateStatus.EFFECTIVE;
    }
    
    public boolean isDueCheck(LocalDateTime lastCheckTime, Long currentRunningHours) {
        if (this.status != TemplateStatus.EFFECTIVE) {
            return false;
        }
        
        if (CheckPeriodType.RUNNING_HOURS.name().equals(this.checkPeriodType)) {
            return currentRunningHours != null && 
                   this.runningHoursThreshold != null && 
                   currentRunningHours >= this.runningHoursThreshold;
        }
        
        if (lastCheckTime == null) {
            return true;
        }
        
        LocalDateTime nextCheckTime = calculateNextCheckTime(lastCheckTime);
        return LocalDateTime.now().isAfter(nextCheckTime) || LocalDateTime.now().isEqual(nextCheckTime);
    }
    
    private LocalDateTime calculateNextCheckTime(LocalDateTime lastCheckTime) {
        if (lastCheckTime == null) {
            return LocalDateTime.now();
        }
        
        int periodValue = this.checkPeriodValue != null ? this.checkPeriodValue : 1;
        String unit = this.checkPeriodUnit != null ? this.checkPeriodUnit : "DAY";
        
        return switch (unit.toUpperCase()) {
            case "HOUR" -> lastCheckTime.plusHours(periodValue);
            case "DAY" -> lastCheckTime.plusDays(periodValue);
            case "WEEK" -> lastCheckTime.plusWeeks(periodValue);
            case "MONTH" -> lastCheckTime.plusMonths(periodValue);
            case "YEAR" -> lastCheckTime.plusYears(periodValue);
            default -> lastCheckTime.plusDays(periodValue);
        };
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTemplateCode() { return templateCode; }
    public void setTemplateCode(String templateCode) { this.templateCode = templateCode; }
    public String getTemplateName() { return templateName; }
    public void setTemplateName(String templateName) { this.templateName = templateName; }
    public String getEquipmentTypeCode() { return equipmentTypeCode; }
    public void setEquipmentTypeCode(String equipmentTypeCode) { this.equipmentTypeCode = equipmentTypeCode; }
    public String getCheckPeriodType() { return checkPeriodType; }
    public void setCheckPeriodType(String checkPeriodType) { this.checkPeriodType = checkPeriodType; }
    public Integer getCheckPeriodValue() { return checkPeriodValue; }
    public void setCheckPeriodValue(Integer checkPeriodValue) { this.checkPeriodValue = checkPeriodValue; }
    public String getCheckPeriodUnit() { return checkPeriodUnit; }
    public void setCheckPeriodUnit(String checkPeriodUnit) { this.checkPeriodUnit = checkPeriodUnit; }
    public Long getRunningHoursThreshold() { return runningHoursThreshold; }
    public void setRunningHoursThreshold(Long runningHoursThreshold) { this.runningHoursThreshold = runningHoursThreshold; }
    public String getAlertBeforeHours() { return alertBeforeHours; }
    public void setAlertBeforeHours(String alertBeforeHours) { this.alertBeforeHours = alertBeforeHours; }
    public TemplateStatus getStatus() { return status; }
    public void setStatus(TemplateStatus status) { this.status = status; }
    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public List<ChecklistItem> getItems() { return items; }
    public void setItems(List<ChecklistItem> items) { this.items = items; }
}