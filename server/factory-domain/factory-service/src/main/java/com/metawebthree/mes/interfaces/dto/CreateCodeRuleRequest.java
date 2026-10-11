package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;

public class CreateCodeRuleRequest {
    
    @NotBlank(message = "规则编码不能为空")
    private String ruleCode;
    
    @NotBlank(message = "规则名称不能为空")
    private String ruleName;
    
    @NotBlank(message = "业务类型不能为空")
    private String businessType;
    
    private String ruleExpression;
    
    @Positive(message = "填充长度必须为正数")
    private Integer paddingLength;
    
    private Long startValue;
    
    private Integer step;
    
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
    public Integer getStep() { return step; }
    public void setStep(Integer step) { this.step = step; }
}