package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

/**
 * 产品SN规则绑定实体
 * 将编码规则绑定到产品，实现产品序列号自动生成
 */
public class ProductSnRule {
    
    private Long id;
    private Long productId;
    private String productCode;
    private Long codeRuleId;
    private String ruleCode;
    private Boolean isActive;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static ProductSnRule create(Long productId, String productCode, 
                                       Long codeRuleId, String ruleCode) {
        ProductSnRule binding = new ProductSnRule();
        binding.productId = productId;
        binding.productCode = productCode;
        binding.codeRuleId = codeRuleId;
        binding.ruleCode = ruleCode;
        binding.isActive = true;
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
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getProductId() { return productId; }
    public void setProductId(Long productId) { this.productId = productId; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public Long getCodeRuleId() { return codeRuleId; }
    public void setCodeRuleId(Long codeRuleId) { this.codeRuleId = codeRuleId; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}