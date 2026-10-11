package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class EquipmentMaintenancePlan {

    
    public enum MaintenanceCycleType {
        TIME_BASED,
        RUNNING_HOURS
    }
    
    private Long id;
    private String planCode;
    private String planName;
    private String description;
    private Long equipmentTypeId;
    private String equipmentTypeCode;
    private MaintenanceCycleType cycleType;
    private Integer cycleDays;
    private Integer cycleRunningHours;
    private Integer advanceAlertDays;
    private Boolean isActive;
    private List<MaintenanceItem> items;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public void create(String planCode, String planName, String equipmentTypeCode, 
                      MaintenanceCycleType cycleType) {
        this.planCode = planCode;
        this.planName = planName;
        this.equipmentTypeCode = equipmentTypeCode;
        this.cycleType = cycleType;
        this.isActive = true;
        this.items = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void setTimeBasedCycle(Integer cycleDays, Integer advanceAlertDays) {
        this.cycleType = MaintenanceCycleType.TIME_BASED;
        this.cycleDays = cycleDays;
        this.advanceAlertDays = advanceAlertDays;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void setRunningHoursCycle(Integer cycleRunningHours, Integer advanceAlertDays) {
        this.cycleType = MaintenanceCycleType.RUNNING_HOURS;
        this.cycleRunningHours = cycleRunningHours;
        this.advanceAlertDays = advanceAlertDays;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addItem(MaintenanceItem item) {
        if (this.items == null) {
            this.items = new ArrayList<>();
        }
        this.items.add(item);
        this.updatedAt = LocalDateTime.now();
    }
    
    public void removeItem(Long itemId) {
        if (this.items == null) {
            return;
        }
        this.items.removeIf(item -> item.getId().equals(itemId));
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }
    
        public static class MaintenanceItem {
        private Long id;
        private Long planId;
        private String itemCode;
        private String itemName;
        private String description;
        private String checkMethod;
        private String standard;
        private Boolean isRequired;
        private Integer sortOrder;
        
        public void create(String itemCode, String itemName) {
            this.itemCode = itemCode;
            this.itemName = itemName;
            this.isRequired = true;
            this.sortOrder = 0;
        }
        
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getPlanId() { return planId; }
        public void setPlanId(Long planId) { this.planId = planId; }
        public String getItemCode() { return itemCode; }
        public void setItemCode(String itemCode) { this.itemCode = itemCode; }
        public String getItemName() { return itemName; }
        public void setItemName(String itemName) { this.itemName = itemName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCheckMethod() { return checkMethod; }
        public void setCheckMethod(String checkMethod) { this.checkMethod = checkMethod; }
        public String getStandard() { return standard; }
        public void setStandard(String standard) { this.standard = standard; }
        public Boolean getIsRequired() { return isRequired; }
        public void setIsRequired(Boolean isRequired) { this.isRequired = isRequired; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPlanCode() { return planCode; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public String getPlanName() { return planName; }
    public void setPlanName(String planName) { this.planName = planName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Long getEquipmentTypeId() { return equipmentTypeId; }
    public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
    public String getEquipmentTypeCode() { return equipmentTypeCode; }
    public void setEquipmentTypeCode(String equipmentTypeCode) { this.equipmentTypeCode = equipmentTypeCode; }
    public MaintenanceCycleType getCycleType() { return cycleType; }
    public void setCycleType(MaintenanceCycleType cycleType) { this.cycleType = cycleType; }
    public Integer getCycleDays() { return cycleDays; }
    public void setCycleDays(Integer cycleDays) { this.cycleDays = cycleDays; }
    public Integer getCycleRunningHours() { return cycleRunningHours; }
    public void setCycleRunningHours(Integer cycleRunningHours) { this.cycleRunningHours = cycleRunningHours; }
    public Integer getAdvanceAlertDays() { return advanceAlertDays; }
    public void setAdvanceAlertDays(Integer advanceAlertDays) { this.advanceAlertDays = advanceAlertDays; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public List<MaintenanceItem> getItems() { return items; }
    public void setItems(List<MaintenanceItem> items) { this.items = items; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}