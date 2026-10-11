package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 工艺参数组模板
 * 用于定义产品类型的标准参数组，关联工艺参数
 */
public class ParameterGroupTemplate {
    
    public enum TemplateStatus {
        DRAFT,      // 草稿
        ACTIVE,     // 生效
        INACTIVE,   // 失效
        ARCHIVED    // 归档
    }
    
    private Long id;
    private String templateCode;        // 模板编码
    private String templateName;        // 模板名称
    private String productType;         // 关联产品类型
    private String description;         // 描述
    private TemplateStatus status;      // 状态
    private Integer displayOrder;       // 显示顺序
    
    // 关联的参数列表（参数ID列表，不直接存储实体）
    private List<Long> parameterIds;
    private List<String> parameterCodes;
    
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;
    
    /**
     * 创建参数组模板
     */
    public void create(String templateCode, String templateName, String productType) {
        this.templateCode = templateCode;
        this.templateName = templateName;
        this.productType = productType;
        this.status = TemplateStatus.DRAFT;
        this.parameterIds = new ArrayList<>();
        this.parameterCodes = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 激活模板
     */
    public void activate() {
        if (this.status == TemplateStatus.DRAFT || this.status == TemplateStatus.INACTIVE) {
            this.status = TemplateStatus.ACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 停用模板
     */
    public void deactivate() {
        if (this.status == TemplateStatus.ACTIVE) {
            this.status = TemplateStatus.INACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 归档模板
     */
    public void archive() {
        if (this.status != TemplateStatus.ARCHIVED) {
            this.status = TemplateStatus.ARCHIVED;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 添加参数到模板
     */
    public void addParameter(Long parameterId, String parameterCode) {
        if (this.parameterIds == null) {
            this.parameterIds = new ArrayList<>();
            this.parameterCodes = new ArrayList<>();
        }
        if (!this.parameterIds.contains(parameterId)) {
            this.parameterIds.add(parameterId);
            this.parameterCodes.add(parameterCode);
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 从模板移除参数
     */
    public void removeParameter(Long parameterId) {
        if (this.parameterIds != null) {
            int index = this.parameterIds.indexOf(parameterId);
            if (index >= 0) {
                this.parameterIds.remove(index);
                if (this.parameterCodes != null && index < this.parameterCodes.size()) {
                    this.parameterCodes.remove(index);
                }
                this.updatedAt = LocalDateTime.now();
            }
        }
    }
    
    /**
     * 验证模板是否有效
     */
    public boolean isValid() {
        return templateCode != null && !templateCode.isEmpty()
            && templateName != null && !templateName.isEmpty()
            && productType != null && !productType.isEmpty();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getTemplateCode() { return templateCode; }
    public void setTemplateCode(String templateCode) { this.templateCode = templateCode; }
    
    public String getTemplateName() { return templateName; }
    public void setTemplateName(String templateName) { this.templateName = templateName; }
    
    public String getProductType() { return productType; }
    public void setProductType(String productType) { this.productType = productType; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public TemplateStatus getStatus() { return status; }
    public void setStatus(TemplateStatus status) { this.status = status; }
    
    public Integer getDisplayOrder() { return displayOrder; }
    public void setDisplayOrder(Integer displayOrder) { this.displayOrder = displayOrder; }
    
    public List<Long> getParameterIds() { return parameterIds; }
    public void setParameterIds(List<Long> parameterIds) { this.parameterIds = parameterIds; }
    
    public List<String> getParameterCodes() { return parameterCodes; }
    public void setParameterCodes(List<String> parameterCodes) { this.parameterCodes = parameterCodes; }
    
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}