package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class MaterialIssueConfig {
    
    private Long id;
    private String configCode;
    private String configName;
    private String workshopId;
    private String productCode;
    private String issueMode;
    private String issueRule;
    private Integer leadTimeHours;
    private Integer bufferHours;
    private Boolean isActive;
    private Integer priority;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum IssueMode {
        PRE_PICKING,
        PULL,
        JIT
    }
    
    public enum IssueRule {
        FIFO,
        LIFO,
        LOCKED_BATCH,
        EXPIRY_FIRST
    }
    
    public static MaterialIssueConfig create(String configCode, String configName,
                                              String workshopId, String issueMode) {
        MaterialIssueConfig config = new MaterialIssueConfig();
        config.configCode = configCode;
        config.configName = configName;
        config.workshopId = workshopId;
        config.issueMode = issueMode;
        config.issueRule = IssueRule.FIFO.name();
        config.leadTimeHours = 0;
        config.bufferHours = 0;
        config.isActive = true;
        config.priority = 0;
        config.createdAt = LocalDateTime.now();
        config.updatedAt = LocalDateTime.now();
        return config;
    }
    
    public void updateIssueMode(String issueMode) {
        this.issueMode = issueMode;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void updateIssueRule(String issueRule) {
        this.issueRule = issueRule;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void configureTiming(Integer leadTimeHours, Integer bufferHours) {
        this.leadTimeHours = leadTimeHours;
        this.bufferHours = bufferHours;
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
    
    public void updatePriority(Integer priority) {
        this.priority = priority;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isPrePickingMode() {
        return IssueMode.PRE_PICKING.name().equals(this.issueMode);
    }
    
    public boolean isPullMode() {
        return IssueMode.PULL.name().equals(this.issueMode);
    }
    
    public boolean isJitMode() {
        return IssueMode.JIT.name().equals(this.issueMode);
    }
    
    public boolean isFifoRule() {
        return IssueRule.FIFO.name().equals(this.issueRule);
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getConfigCode() { return configCode; }
    public void setConfigCode(String configCode) { this.configCode = configCode; }
    public String getConfigName() { return configName; }
    public void setConfigName(String configName) { this.configName = configName; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getIssueMode() { return issueMode; }
    public void setIssueMode(String issueMode) { this.issueMode = issueMode; }
    public String getIssueRule() { return issueRule; }
    public void setIssueRule(String issueRule) { this.issueRule = issueRule; }
    public Integer getLeadTimeHours() { return leadTimeHours; }
    public void setLeadTimeHours(Integer leadTimeHours) { this.leadTimeHours = leadTimeHours; }
    public Integer getBufferHours() { return bufferHours; }
    public void setBufferHours(Integer bufferHours) { this.bufferHours = bufferHours; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}