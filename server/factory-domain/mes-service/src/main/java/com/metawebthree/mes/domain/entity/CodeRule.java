package com.metawebthree.mes.domain.entity;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

public class CodeRule {
    
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String businessType;
    private String ruleExpression;
    private Long startValue;
    private Long currentValue;
    private Integer step;
    private Integer paddingLength;
    private List<RuleElement> elements;
    private RuleStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private static final Map<String, DateTimeFormatter> DATE_FORMATTERS = new HashMap<>();
    private static final Random RANDOM = new Random();
    private static final String NUMERIC_CHARS = "0123456789";
    private static final String ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static {
        DATE_FORMATTERS.put("YYYY", DateTimeFormatter.ofPattern("yyyy"));
        DATE_FORMATTERS.put("YY", DateTimeFormatter.ofPattern("yy"));
        DATE_FORMATTERS.put("MM", DateTimeFormatter.ofPattern("MM"));
        DATE_FORMATTERS.put("DD", DateTimeFormatter.ofPattern("dd"));
    }
    
    public enum RuleStatus {
        ACTIVE, INACTIVE
    }
    
    // SN生成专用元素类型
    public enum SnElementType {
        RANDOM_NUMERIC,      // 随机数字
        RANDOM_ALPHANUMERIC, // 随机字母数字
        CHECKSUM_MOD10,     // Mod10 校验位
        CHECKSUM_MOD11,     // Mod11 校验位
        UUID_SHORT          // 短UUID(无横线)
    }
    
    public static class RuleElement {
        
        private Long id;
        private ElementType type;
        private String value;
        private String fieldName;
        private Integer sortOrder;
        
        public enum ElementType {
            PREFIX, DATE, SEQUENCE, BUSINESS_FIELD, DELIMITER,
            // SN生成专用类型
            RANDOM_NUMERIC, RANDOM_ALPHANUMERIC, CHECKSUM_MOD10, CHECKSUM_MOD11, UUID_SHORT
        }
        
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public ElementType getType() { return type; }
        public void setType(ElementType type) { this.type = type; }
        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
        public String getFieldName() { return fieldName; }
        public void setFieldName(String fieldName) { this.fieldName = fieldName; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    }
    
    public static CodeRule create(String ruleCode, String ruleName, String businessType,
                                  String ruleExpression, Integer paddingLength) {
        CodeRule rule = new CodeRule();
        rule.ruleCode = ruleCode;
        rule.ruleName = ruleName;
        rule.businessType = businessType;
        rule.ruleExpression = ruleExpression;
        rule.paddingLength = paddingLength;
        rule.startValue = 1L;
        rule.currentValue = 1L;
        rule.step = 1;
        rule.elements = new ArrayList<>();
        rule.status = RuleStatus.ACTIVE;
        rule.createdAt = LocalDateTime.now();
        rule.updatedAt = LocalDateTime.now();
        return rule;
    }
    
    public RuleElement addElement(RuleElement.ElementType type, String value) {
        RuleElement element = new RuleElement();
        element.setType(type);
        element.setValue(value);
        this.elements.add(element);
        return element;
    }
    
    public String peekNextCode() {
        Long seq = this.currentValue;
        
        StringBuilder sb = new StringBuilder();
        StringBuilder snBuilder = new StringBuilder();
        
        // 第一遍：收集非校验位元素
        for (RuleElement element : this.elements) {
            switch (element.getType()) {
                case PREFIX -> sb.append(element.getValue());
                case DATE -> sb.append(formatDate(element.getValue()));
                case SEQUENCE -> sb.append(String.format("%0" + paddingLength + "d", seq));
                case BUSINESS_FIELD -> sb.append("{").append(element.getFieldName()).append("}");
                case DELIMITER -> sb.append(element.getValue());
                case RANDOM_NUMERIC -> {
                    int len = element.getValue() != null ? Integer.parseInt(element.getValue()) : 6;
                    sb.append(generateRandomNumeric(len));
                }
                case RANDOM_ALPHANUMERIC -> {
                    int len = element.getValue() != null ? Integer.parseInt(element.getValue()) : 8;
                    sb.append(generateRandomAlphanumeric(len));
                }
                case UUID_SHORT -> sb.append(UUID.randomUUID().toString().replace("-", "").substring(0, 12));
                case CHECKSUM_MOD10, CHECKSUM_MOD11 -> {
                    // 暂存，稍后计算校验位
                }
            }
        }
        
        // 第二遍：处理校验位（基于第一遍的结果）
        snBuilder.setLength(0);
        for (RuleElement element : this.elements) {
            switch (element.getType()) {
                case CHECKSUM_MOD10 -> snBuilder.append(calculateMod10(sb.toString()));
                case CHECKSUM_MOD11 -> snBuilder.append(calculateMod11(sb.toString()));
                default -> {
                    // 其他类型已处理
                }
            }
        }
        
        return sb.toString() + snBuilder.toString();
    }
    
    public void advanceSequence() {
        this.currentValue += this.step;
    }
    
    @Deprecated
    public String generateNextCode() {
        String code = peekNextCode();
        advanceSequence();
        return code;
    }
    
    private String formatDate(String pattern) {
        LocalDate now = LocalDate.now();
        String result = pattern;
        for (Map.Entry<String, DateTimeFormatter> entry : DATE_FORMATTERS.entrySet()) {
            if (result.contains(entry.getKey())) {
                result = result.replace(entry.getKey(), now.format(entry.getValue()));
            }
        }
        return result;
    }
    
    private String generateRandomNumeric(int length) {
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            sb.append(NUMERIC_CHARS.charAt(RANDOM.nextInt(NUMERIC_CHARS.length())));
        }
        return sb.toString();
    }
    
    private String generateRandomAlphanumeric(int length) {
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            sb.append(ALPHANUMERIC_CHARS.charAt(RANDOM.nextInt(ALPHANUMERIC_CHARS.length())));
        }
        return sb.toString();
    }
    
    private char calculateMod10(String input) {
        int sum = 0;
        boolean alternate = true;
        for (int i = input.length() - 1; i >= 0; i--) {
            int n = Character.getNumericValue(input.charAt(i));
            if (alternate) {
                n *= 2;
                if (n > 9) n -= 9;
            }
            sum += n;
            alternate = !alternate;
        }
        int checkDigit = (10 - (sum % 10)) % 10;
        return Character.forDigit(checkDigit, 10);
    }
    
    private char calculateMod11(String input) {
        int[] weights = {2, 3, 4, 5, 6, 7, 8, 9, 10};
        int sum = 0;
        int len = input.length();
        for (int i = 0; i < len; i++) {
            int n = Character.getNumericValue(input.charAt(len - 1 - i));
            int weight = weights[i % weights.length];
            sum += n * weight;
        }
        int remainder = sum % 11;
        int checkDigit = 11 - remainder;
        if (checkDigit == 10) return 'X';
        if (checkDigit == 11) return '0';
        return Character.forDigit(checkDigit, 10);
    }
    
    public void resetSequence() {
        this.currentValue = this.startValue;
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