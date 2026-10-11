package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.LocalDateTime;

@TableName("mes_qc_inspection_plan")
public class QcInspectionPlanDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    private String planCode;
    private String planName;
    private String inspectionType;
    private String applicableProductTypes;
    private Integer version;
    private String samplingPlanCode;
    private String samplingType;
    private String aql;
    private String inspectionLevel;
    private Integer sampleSize;
    private String acceptNumber;
    private String rejectNumber;
    private String dispositionRule;
    private String qualifiedFlow;
    private String unqualifiedFlow;
    private String specialApprovalFlow;
    private String status;
    private LocalDateTime effectiveDate;
    private LocalDateTime expirationDate;
    private Integer sortOrder;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPlanCode() { return planCode; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public String getPlanName() { return planName; }
    public void setPlanName(String planName) { this.planName = planName; }
    public String getInspectionType() { return inspectionType; }
    public void setInspectionType(String inspectionType) { this.inspectionType = inspectionType; }
    public String getApplicableProductTypes() { return applicableProductTypes; }
    public void setApplicableProductTypes(String applicableProductTypes) { this.applicableProductTypes = applicableProductTypes; }
    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }
    public String getSamplingPlanCode() { return samplingPlanCode; }
    public void setSamplingPlanCode(String samplingPlanCode) { this.samplingPlanCode = samplingPlanCode; }
    public String getSamplingType() { return samplingType; }
    public void setSamplingType(String samplingType) { this.samplingType = samplingType; }
    public String getAql() { return aql; }
    public void setAql(String aql) { this.aql = aql; }
    public String getInspectionLevel() { return inspectionLevel; }
    public void setInspectionLevel(String inspectionLevel) { this.inspectionLevel = inspectionLevel; }
    public Integer getSampleSize() { return sampleSize; }
    public void setSampleSize(Integer sampleSize) { this.sampleSize = sampleSize; }
    public String getAcceptNumber() { return acceptNumber; }
    public void setAcceptNumber(String acceptNumber) { this.acceptNumber = acceptNumber; }
    public String getRejectNumber() { return rejectNumber; }
    public void setRejectNumber(String rejectNumber) { this.rejectNumber = rejectNumber; }
    public String getDispositionRule() { return dispositionRule; }
    public void setDispositionRule(String dispositionRule) { this.dispositionRule = dispositionRule; }
    public String getQualifiedFlow() { return qualifiedFlow; }
    public void setQualifiedFlow(String qualifiedFlow) { this.qualifiedFlow = qualifiedFlow; }
    public String getUnqualifiedFlow() { return unqualifiedFlow; }
    public void setUnqualifiedFlow(String unqualifiedFlow) { this.unqualifiedFlow = unqualifiedFlow; }
    public String getSpecialApprovalFlow() { return specialApprovalFlow; }
    public void setSpecialApprovalFlow(String specialApprovalFlow) { this.specialApprovalFlow = specialApprovalFlow; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
    public LocalDateTime getExpirationDate() { return expirationDate; }
    public void setExpirationDate(LocalDateTime expirationDate) { this.expirationDate = expirationDate; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}