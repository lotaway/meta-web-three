package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class QcInspectionType {
    
    private Long id;
    private String typeCode;
    private String typeName;
    private InspectionCategory category;
    private String description;
    private String applicableProducts;
    private String defaultSamplingPlan;
    private String defaultAql;
    private Integer defaultTimeoutHours;
    private Boolean requireCertificate;
    private Boolean requireTestReport;
    private InspectionStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum InspectionCategory {
        IQC,
        IPQC,
        FQC,
        OQC
    }
    
    public enum InspectionStatus {
        ACTIVE, INACTIVE
    }
    
    public static QcInspectionType create(String typeCode, String typeName, 
            InspectionCategory category) {
        QcInspectionType type = new QcInspectionType();
        type.typeCode = typeCode;
        type.typeName = typeName;
        type.category = category;
        type.status = InspectionStatus.ACTIVE;
        type.requireCertificate = false;
        type.requireTestReport = false;
        type.defaultTimeoutHours = 24;
        type.sortOrder = 0;
        type.createdAt = LocalDateTime.now();
        type.updatedAt = LocalDateTime.now();
        return type;
    }
    
    public void update(String typeName, InspectionCategory category, String description,
            String applicableProducts, String defaultSamplingPlan, String defaultAql,
            Integer defaultTimeoutHours, Boolean requireCertificate, Boolean requireTestReport) {
        this.typeName = typeName;
        this.category = category;
        this.description = description;
        this.applicableProducts = applicableProducts;
        this.defaultSamplingPlan = defaultSamplingPlan;
        this.defaultAql = defaultAql;
        this.defaultTimeoutHours = defaultTimeoutHours;
        this.requireCertificate = requireCertificate;
        this.requireTestReport = requireTestReport;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = InspectionStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = InspectionStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isActive() {
        return this.status == InspectionStatus.ACTIVE;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public InspectionCategory getCategory() { return category; }
    public void setCategory(InspectionCategory category) { this.category = category; }
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
    public InspectionStatus getStatus() { return status; }
    public void setStatus(InspectionStatus status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}