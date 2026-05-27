package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

/**
 * 设备点检模板-点检项关联
 */
public class TemplateChecklistItem {
    
    private Long id;
    private Long templateId;
    private Long itemId;
    private Integer itemSequence;
    private Boolean isMandatory;
    private String defaultValue;
    private String abnormalJudgment;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private ChecklistItem checklistItem;
    
    public static TemplateChecklistItem create(Long templateId, Long itemId, Integer itemSequence) {
        TemplateChecklistItem binding = new TemplateChecklistItem();
        binding.templateId = templateId;
        binding.itemId = itemId;
        binding.itemSequence = itemSequence;
        binding.isMandatory = true;
        binding.sortOrder = itemSequence;
        binding.createdAt = LocalDateTime.now();
        binding.updatedAt = LocalDateTime.now();
        return binding;
    }
    
    public void update(Integer itemSequence, Boolean isMandatory, String defaultValue, String abnormalJudgment) {
        this.itemSequence = itemSequence;
        this.isMandatory = isMandatory;
        this.defaultValue = defaultValue;
        this.abnormalJudgment = abnormalJudgment;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getTemplateId() { return templateId; }
    public void setTemplateId(Long templateId) { this.templateId = templateId; }
    public Long getItemId() { return itemId; }
    public void setItemId(Long itemId) { this.itemId = itemId; }
    public Integer getItemSequence() { return itemSequence; }
    public void setItemSequence(Integer itemSequence) { this.itemSequence = itemSequence; }
    public Boolean getIsMandatory() { return isMandatory; }
    public void setIsMandatory(Boolean isMandatory) { this.isMandatory = isMandatory; }
    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
    public String getAbnormalJudgment() { return abnormalJudgment; }
    public void setAbnormalJudgment(String abnormalJudgment) { this.abnormalJudgment = abnormalJudgment; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public ChecklistItem getChecklistItem() { return checklistItem; }
    public void setChecklistItem(ChecklistItem checklistItem) { this.checklistItem = checklistItem; }
}