package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

public class CodeRuleDTO {
    
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String businessType;
    private String ruleExpression;
    private Integer paddingLength;
    private Long startValue;
    private Long currentValue;
    private Integer step;
    private String status;
    private List<RuleElementDTO> elements;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static CodeRuleDTO fromEntity(
            com.metawebthree.mes.domain.entity.CodeRule entity) {
        if (entity == null) return null;
        
        CodeRuleDTO dto = new CodeRuleDTO();
        dto.setId(entity.getId());
        dto.setRuleCode(entity.getRuleCode());
        dto.setRuleName(entity.getRuleName());
        dto.setBusinessType(entity.getBusinessType());
        dto.setRuleExpression(entity.getRuleExpression());
        dto.setPaddingLength(entity.getPaddingLength());
        dto.setStartValue(entity.getStartValue());
        dto.setCurrentValue(entity.getCurrentValue());
        dto.setStep(entity.getStep());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getElements() != null) {
            dto.setElements(entity.getElements().stream()
                    .map(RuleElementDTO::fromEntity)
                    .collect(Collectors.toList()));
        }
        
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public String getBusinessType() { return businessType; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public String getRuleExpression() { return ruleExpression; }
    public void setRuleExpression(String ruleExpression) { this.ruleExpression = ruleExpression; }
    public Integer getPaddingLength() { return paddingLength; }
    public void setPaddingLength(Integer paddingLength) { this.paddingLength = paddingLength; }
    public Long getStartValue() { return startValue; }
    public void setStartValue(Long startValue) { this.startValue = startValue; }
    public Long getCurrentValue() { return currentValue; }
    public void setCurrentValue(Long currentValue) { this.currentValue = currentValue; }
    public Integer getStep() { return step; }
    public void setStep(Integer step) { this.step = step; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public List<RuleElementDTO> getElements() { return elements; }
    public void setElements(List<RuleElementDTO> elements) { this.elements = elements; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public static class RuleElementDTO {
        private Long id;
        private String elementType;
        private String elementValue;
        private String fieldName;
        private Integer sortOrder;
        
        public static RuleElementDTO fromEntity(
                com.metawebthree.mes.domain.entity.CodeRule.RuleElement element) {
            if (element == null) return null;
            
            RuleElementDTO dto = new RuleElementDTO();
            dto.setId(element.getId());
            dto.setElementType(element.getType() != null ? element.getType().name() : null);
            dto.setElementValue(element.getValue());
            dto.setFieldName(element.getFieldName());
            dto.setSortOrder(element.getSortOrder());
            return dto;
        }
        
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getElementType() { return elementType; }
        public void setElementType(String elementType) { this.elementType = elementType; }
        public String getElementValue() { return elementValue; }
        public void setElementValue(String elementValue) { this.elementValue = elementValue; }
        public String getFieldName() { return fieldName; }
        public void setFieldName(String fieldName) { this.fieldName = fieldName; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    }
}