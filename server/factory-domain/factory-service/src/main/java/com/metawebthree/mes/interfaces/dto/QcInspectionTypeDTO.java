package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.QcInspectionType;
import java.time.LocalDateTime;

public class QcInspectionTypeDTO {
    
    private Long id;
    private String typeCode;
    private String typeName;
    private String category;
    private String categoryName;
    private String description;
    private String applicableProducts;
    private String defaultSamplingPlan;
    private String defaultAql;
    private Integer defaultTimeoutHours;
    private Boolean requireCertificate;
    private Boolean requireTestReport;
    private String status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static QcInspectionTypeDTO fromEntity(QcInspectionType entity) {
        if (entity == null) return null;
        
        QcInspectionTypeDTO dto = new QcInspectionTypeDTO();
        dto.setId(entity.getId());
        dto.setTypeCode(entity.getTypeCode());
        dto.setTypeName(entity.getTypeName());
        dto.setCategory(entity.getCategory() != null ? entity.getCategory().name() : null);
        dto.setCategoryName(getCategoryName(entity.getCategory()));
        dto.setDescription(entity.getDescription());
        dto.setApplicableProducts(entity.getApplicableProducts());
        dto.setDefaultSamplingPlan(entity.getDefaultSamplingPlan());
        dto.setDefaultAql(entity.getDefaultAql());
        dto.setDefaultTimeoutHours(entity.getDefaultTimeoutHours());
        dto.setRequireCertificate(entity.getRequireCertificate());
        dto.setRequireTestReport(entity.getRequireTestReport());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setSortOrder(entity.getSortOrder());
        return dto;
    }
    
    public QcInspectionType toEntity() {
        QcInspectionType entity = new QcInspectionType();
        entity.setId(this.id);
        entity.setTypeCode(this.typeCode);
        entity.setTypeName(this.typeName);
        entity.setCategory(this.category != null ? QcInspectionType.InspectionCategory.valueOf(this.category) : null);
        entity.setDescription(this.description);
        entity.setApplicableProducts(this.applicableProducts);
        entity.setDefaultSamplingPlan(this.defaultSamplingPlan);
        entity.setDefaultAql(this.defaultAql);
        entity.setDefaultTimeoutHours(this.defaultTimeoutHours);
        entity.setRequireCertificate(this.requireCertificate);
        entity.setRequireTestReport(this.requireTestReport);
        entity.setStatus(this.status != null ? QcInspectionType.InspectionStatus.valueOf(this.status) : null);
        entity.setSortOrder(this.sortOrder);
        return entity;
    }
    
    private static String getCategoryName(QcInspectionType.InspectionCategory category) {
        if (category == null) return null;
        switch (category) {
            case IQC: return "来料检验";
            case IPQC: return "制程检验";
            case FQC: return "最终检验";
            case OQC: return "出货检验";
            default: return category.name();
        }
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public String getCategoryName() { return categoryName; }
    public void setCategoryName(String categoryName) { this.categoryName = categoryName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getApplicableProducts() { return applicableProducts; }
    public void setApplicableProducts(String applicableProducts) { this.applicableProducts = applicableProducts; }
    public String getDefaultSamplingPlan() { return defaultSamplingPlan; }
    public void setDefaultSamplingPlan(String defaultSamplingPlan) { this.defaultSamplingPlan = defaultSamplingPlan; }
    public String getDefaultAql() { return defaultAql; }
    public void setDefaultAql(String defaultAql) { this.defaultAql = defaultAql; }
    public Integer getDefaultTimeoutHours() { return defaultTimeoutHours; }
    public void setDefaultTimeoutHours(Integer defaultTimeoutHours) { this.defaultTimeoutHours = defaultTimeoutHours; }
    public Boolean getRequireCertificate() { return requireCertificate; }
    public void setRequireCertificate(Boolean requireCertificate) { this.requireCertificate = requireCertificate; }
    public Boolean getRequireTestReport() { return requireTestReport; }
    public void setRequireTestReport(Boolean requireTestReport) { this.requireTestReport = requireTestReport; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}