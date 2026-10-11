package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.QcInspectionPlan;
import java.time.LocalDateTime;
import java.util.List;

public class QcInspectionPlanDTO {
    
    private Long id;
    private String planCode;
    private String planName;
    private String inspectionType;
    private String inspectionTypeName;
    private String applicableProductTypes;
    private Integer version;
    private String samplingPlanCode;
    private String samplingType;
    private String samplingTypeName;
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
    private String statusName;
    private LocalDateTime effectiveDate;
    private LocalDateTime expirationDate;
    private Integer sortOrder;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<QcPlanItemDTO> planItems;
    
    public static QcInspectionPlanDTO fromEntity(QcInspectionPlan entity) {
        if (entity == null) return null;
        
        QcInspectionPlanDTO dto = new QcInspectionPlanDTO();
        dto.setId(entity.getId());
        dto.setPlanCode(entity.getPlanCode());
        dto.setPlanName(entity.getPlanName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionTypeName(getInspectionTypeName(entity.getInspectionType()));
        dto.setApplicableProductTypes(entity.getApplicableProductTypes());
        dto.setVersion(entity.getVersion());
        dto.setSamplingPlanCode(entity.getSamplingPlanCode());
        dto.setSamplingType(entity.getSamplingType());
        dto.setSamplingTypeName(getSamplingTypeName(entity.getSamplingType()));
        dto.setAql(entity.getAql());
        dto.setInspectionLevel(entity.getInspectionLevel());
        dto.setSampleSize(entity.getSampleSize());
        dto.setAcceptNumber(entity.getAcceptNumber());
        dto.setRejectNumber(entity.getRejectNumber());
        dto.setDispositionRule(entity.getDispositionRule());
        dto.setQualifiedFlow(entity.getQualifiedFlow());
        dto.setUnqualifiedFlow(entity.getUnqualifiedFlow());
        dto.setSpecialApprovalFlow(entity.getSpecialApprovalFlow());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setStatusName(getStatusName(entity.getStatus()));
        dto.setEffectiveDate(entity.getEffectiveDate());
        dto.setExpirationDate(entity.getExpirationDate());
        dto.setSortOrder(entity.getSortOrder());
        dto.setRemark(entity.getRemark());
        return dto;
    }
    
    public QcInspectionPlan toEntity() {
        QcInspectionPlan entity = new QcInspectionPlan();
        entity.setId(this.id);
        entity.setPlanCode(this.planCode);
        entity.setPlanName(this.planName);
        entity.setInspectionType(this.inspectionType);
        entity.setApplicableProductTypes(this.applicableProductTypes);
        entity.setVersion(this.version);
        entity.setSamplingPlanCode(this.samplingPlanCode);
        entity.setSamplingType(this.samplingType);
        entity.setAql(this.aql);
        entity.setInspectionLevel(this.inspectionLevel);
        entity.setSampleSize(this.sampleSize);
        entity.setAcceptNumber(this.acceptNumber);
        entity.setRejectNumber(this.rejectNumber);
        entity.setDispositionRule(this.dispositionRule);
        entity.setQualifiedFlow(this.qualifiedFlow);
        entity.setUnqualifiedFlow(this.unqualifiedFlow);
        entity.setSpecialApprovalFlow(this.specialApprovalFlow);
        entity.setStatus(this.status != null ? QcInspectionPlan.PlanStatus.valueOf(this.status) : null);
        entity.setEffectiveDate(this.effectiveDate);
        entity.setExpirationDate(this.expirationDate);
        entity.setSortOrder(this.sortOrder);
        entity.setRemark(this.remark);
        return entity;
    }
    
    private static String getInspectionTypeName(String type) {
        if (type == null) return null;
        switch (type) {
            case "IQC": return "来料检验";
            case "IPQC": return "制程检验";
            case "FQC": return "最终检验";
            case "OQC": return "出货检验";
            default: return type;
        }
    }
    
    private static String getSamplingTypeName(String type) {
        if (type == null) return null;
        switch (type) {
            case "FULL_INSPECTION": return "全检";
            case "RANDOM_SAMPLING": return "随机抽样";
            case "SYSTEMATIC_SAMPLING": return "系统抽样";
            case "DOUBLE_SAMPLING": return "二次抽样";
            default: return type;
        }
    }
    
    private static String getStatusName(QcInspectionPlan.PlanStatus status) {
        if (status == null) return null;
        switch (status) {
            case DRAFT: return "草稿";
            case EFFECTIVE: return "生效";
            case EXPIRED: return "已过期";
            case CANCELLED: return "已取消";
            default: return status.name();
        }
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPlanCode() { return planCode; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public String getPlanName() { return planName; }
    public void setPlanName(String planName) { this.planName = planName; }
    public String getInspectionType() { return inspectionType; }
    public void setInspectionType(String inspectionType) { this.inspectionType = inspectionType; }
    public String getInspectionTypeName() { return inspectionTypeName; }
    public void setInspectionTypeName(String inspectionTypeName) { this.inspectionTypeName = inspectionTypeName; }
    public String getApplicableProductTypes() { return applicableProductTypes; }
    public void setApplicableProductTypes(String applicableProductTypes) { this.applicableProductTypes = applicableProductTypes; }
    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }
    public String getSamplingPlanCode() { return samplingPlanCode; }
    public void setSamplingPlanCode(String samplingPlanCode) { this.samplingPlanCode = samplingPlanCode; }
    public String getSamplingType() { return samplingType; }
    public void setSamplingType(String samplingType) { this.samplingType = samplingType; }
    public String getSamplingTypeName() { return samplingTypeName; }
    public void setSamplingTypeName(String samplingTypeName) { this.samplingTypeName = samplingTypeName; }
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
    public String getStatusName() { return statusName; }
    public void setStatusName(String statusName) { this.statusName = statusName; }
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
    public List<QcPlanItemDTO> getPlanItems() { return planItems; }
    public void setPlanItems(List<QcPlanItemDTO> planItems) { this.planItems = planItems; }
}