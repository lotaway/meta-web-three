package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 工艺参数组模板 DTO
 */
public class ParameterGroupTemplateDTO {
    
    private Long id;
    private String templateCode;
    private String templateName;
    private String productType;
    private String description;
    private String status;
    private Integer displayOrder;
    private List<Long> parameterIds;
    private List<String> parameterCodes;
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;
    
    /**
     * 创建请求
     */
    public static class CreateRequest {
        private String templateCode;
        private String templateName;
        private String productType;
        private String description;
        private Integer displayOrder;
        private List<Long> parameterIds;
        
        public String getTemplateCode() { return templateCode; }
        public void setTemplateCode(String templateCode) { this.templateCode = templateCode; }
        public String getTemplateName() { return templateName; }
        public void setTemplateName(String templateName) { this.templateName = templateName; }
        public String getProductType() { return productType; }
        public void setProductType(String productType) { this.productType = productType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getDisplayOrder() { return displayOrder; }
        public void setDisplayOrder(Integer displayOrder) { this.displayOrder = displayOrder; }
        public List<Long> getParameterIds() { return parameterIds; }
        public void setParameterIds(List<Long> parameterIds) { this.parameterIds = parameterIds; }
        
        public com.metawebthree.mes.domain.entity.ParameterGroupTemplate toEntity() {
            com.metawebthree.mes.domain.entity.ParameterGroupTemplate template = 
                new com.metawebthree.mes.domain.entity.ParameterGroupTemplate();
            template.create(templateCode, templateName, productType);
            template.setDescription(description);
            template.setDisplayOrder(displayOrder);
            template.setParameterIds(parameterIds);
            return template;
        }
    }
    
    /**
     * 更新请求
     */
    public static class UpdateRequest {
        private String templateName;
        private String productType;
        private String description;
        private Integer displayOrder;
        private List<Long> parameterIds;
        
        public String getTemplateName() { return templateName; }
        public void setTemplateName(String templateName) { this.templateName = templateName; }
        public String getProductType() { return productType; }
        public void setProductType(String productType) { this.productType = productType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getDisplayOrder() { return displayOrder; }
        public void setDisplayOrder(Integer displayOrder) { this.displayOrder = displayOrder; }
        public List<Long> getParameterIds() { return parameterIds; }
        public void setParameterIds(List<Long> parameterIds) { this.parameterIds = parameterIds; }
    }
    
    /**
     * 从实体转换
     */
    public static ParameterGroupTemplateDTO fromEntity(
            com.metawebthree.mes.domain.entity.ParameterGroupTemplate entity) {
        if (entity == null) return null;
        
        ParameterGroupTemplateDTO dto = new ParameterGroupTemplateDTO();
        dto.setId(entity.getId());
        dto.setTemplateCode(entity.getTemplateCode());
        dto.setTemplateName(entity.getTemplateName());
        dto.setProductType(entity.getProductType());
        dto.setDescription(entity.getDescription());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setDisplayOrder(entity.getDisplayOrder());
        dto.setParameterIds(entity.getParameterIds());
        dto.setParameterCodes(entity.getParameterCodes());
        dto.setCreatedBy(entity.getCreatedBy());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedBy(entity.getUpdatedBy());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
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
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
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