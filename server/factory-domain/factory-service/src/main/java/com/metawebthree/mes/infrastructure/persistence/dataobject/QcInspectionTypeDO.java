package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.LocalDateTime;

@TableName("mes_qc_inspection_type")
public class QcInspectionTypeDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    private String typeCode;
    private String typeName;
    private String category;
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
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
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