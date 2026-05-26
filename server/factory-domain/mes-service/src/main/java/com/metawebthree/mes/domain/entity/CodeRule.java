package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 编码规则配置
 * 用于实现配置化MES的编码规则配置
 */
public class CodeRule {
    
    private Long id;
    
    /**
     * 规则编码（唯一标识）
     */
    private String ruleCode;
    
    /**
     * 规则名称
     */
    private String ruleName;
    
    /**
     * 适用业务类型: WORK_ORDER, PRODUCTION_TASK, MATERIAL, EQUIPMENT, QC_INSPECTION, PRODUCT_SN
     */
    private String businessType;
    
    /**
     * 规则表达式
     * 示例: "MO-[工厂代码]-[YYYYMMDD]-[流水号4位]"
     */
    private String ruleExpression;
    
    /**
     * 起始值
     */
    private Long startValue;
    
    /**
     * 当前值
     */
    private Long currentValue;
    
    /**
     * 步长
     */
    private Integer step;
    
    /**
     * 填充长度（如4位流水号: 0001）
     */
    private Integer paddingLength;
    
    /**
     * 规则要素列表
     */
    private List<RuleElement> elements;
    
    /**
     * 状态: ACTIVE, INACTIVE
     */
    private RuleStatus status;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
    
    public enum RuleStatus {
        ACTIVE, INACTIVE
    }
    
    /**
     * 编码规则要素
     */
    public static class RuleElement {
        
        /**
         * 要素类型: PREFIX, DATE, SEQUENCE, BUSINESS_FIELD, DELIMITER
         */
        private ElementType type;
        
        /**
         * 要素值或表达式
         */
        private String value;
        
        /**
         * 业务字段名（当type为BUSINESS_FIELD时）
         */
        private String fieldName;
        
        public enum ElementType {
            PREFIX,      // 固定前缀
            DATE,        // 日期占位符
            SEQUENCE,    // 流水号
            BUSINESS_FIELD, // 业务字段引用
            DELIMITER    // 分隔符
        }
        
        // Getters and Setters
        public ElementType getType() { return type; }
        public void setType(ElementType type) { this.type = type; }
        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
        public String getFieldName() { return fieldName; }
        public void setFieldName(String fieldName) { this.fieldName = fieldName; }
    }
    
    /**
     * 创建编码规则
     */
    public void create(String ruleCode, String ruleName, String businessType, 
                       String ruleExpression, Integer paddingLength) {
        this.ruleCode = ruleCode;
        this.ruleName = ruleName;
        this.businessType = businessType;
        this.ruleExpression = ruleExpression;
        this.paddingLength = paddingLength;
        this.startValue = 1L;
        this.currentValue = 1L;
        this.step = 1;
        this.elements = new ArrayList<>();
        this.status = RuleStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加规则要素
     */
    public RuleElement addElement(RuleElement.ElementType type, String value) {
        RuleElement element = new RuleElement();
        element.setType(type);
        element.setValue(value);
        this.elements.add(element);
        return element;
    }
    
    /**
     * 生成下一个编码
     */
    public String generateNextCode() {
        Long seq = this.currentValue;
        this.currentValue += this.step;
        
        StringBuilder sb = new StringBuilder();
        for (RuleElement element : this.elements) {
            switch (element.getType()) {
                case PREFIX:
                    sb.append(element.getValue());
                    break;
                case DATE:
                    sb.append(formatDate(element.getValue()));
                    break;
                case SEQUENCE:
                    sb.append(String.format("%0" + paddingLength + "d", seq));
                    break;
                case BUSINESS_FIELD:
                    // 业务字段需要在运行时替换
                    sb.append("{").append(element.getFieldName()).append("}");
                    break;
                case DELIMITER:
                    sb.append(element.getValue());
                    break;
            }
        }
        return sb.toString();
    }
    
    /**
     * 格式化日期
     */
    private String formatDate(String pattern) {
        java.time.LocalDate now = java.time.LocalDate.now();
        String result = pattern;
        result = result.replace("YYYY", String.valueOf(now.getYear()));
        result = result.replace("YY", String.format("%02d", now.getYear() % 100));
        result = result.replace("MM", String.format("%02d", now.getMonthValue()));
        result = result.replace("DD", String.format("%02d", now.getDayOfMonth()));
        return result;
    }
    
    /**
     * 重置流水号
     */
    public void resetSequence() {
        this.currentValue = this.startValue;
    }
    
    // Getters and Setters
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
    public Long getStartValue() { return startValue; }
    public void setStartValue(Long startValue) { this.startValue = startValue; }
    public Long getCurrentValue() { return currentValue; }
    public void setCurrentValue(Long currentValue) { this.currentValue = currentValue; }
    public Integer getStep() { return step; }
    public void setStep(Integer step) { this.step = step; }
    public Integer getPaddingLength() { return paddingLength; }
    public void setPaddingLength(Integer paddingLength) { this.paddingLength = paddingLength; }
    public List<RuleElement> getElements() { return elements; }
    public void setElements(List<RuleElement> elements) { this.elements = elements; }
    public RuleStatus getStatus() { return status; }
    public void setStatus(RuleStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}