package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class WorkOrderCodeRule {
    
    private Long id;
    private String workshopId;
    private String workOrderType;
    private Long codeRuleId;
    private String ruleCode;
    private Boolean isActive;
    private String description;
    private Integer priority;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static WorkOrderCodeRule create(String workshopId, String workOrderType,
                                           Long codeRuleId, String ruleCode) {
        WorkOrderCodeRule binding = new WorkOrderCodeRule();
        binding.workshopId = workshopId;
        binding.workOrderType = workOrderType;
        binding.codeRuleId = codeRuleId;
        binding.ruleCode = ruleCode;
        binding.isActive = true;
        binding.priority = 0;
        binding.createdAt = LocalDateTime.now();
        binding.updatedAt = LocalDateTime.now();
        return binding;
    }
    
    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void updatePriority(Integer priority) {
        this.priority = priority;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getWorkOrderType() { return workOrderType; }
    public void setWorkOrderType(String workOrderType) { this.workOrderType = workOrderType; }
    public Long getCodeRuleId() { return codeRuleId; }
    public void setCodeRuleId(Long codeRuleId) { this.codeRuleId = codeRuleId; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}