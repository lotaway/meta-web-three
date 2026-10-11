package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.QcConstants;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class QcInspectionPlan {
    
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
    private PlanStatus status;
    private LocalDateTime effectiveDate;
    private LocalDateTime expirationDate;
    private Integer sortOrder;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<QcPlanItem> planItems = new ArrayList<>();
    
    public enum PlanStatus {
        DRAFT, EFFECTIVE, EXPIRED, CANCELLED
    }
    
    public enum SamplingType {
        FULL_INSPECTION, RANDOM_SAMPLING, SYSTEMATIC_SAMPLING, DOUBLE_SAMPLING
    }
    
    public static QcInspectionPlan create(String planCode, String planName, String inspectionType) {
        QcInspectionPlan plan = new QcInspectionPlan();
        plan.planCode = planCode;
        plan.planName = planName;
        plan.inspectionType = inspectionType;
        plan.version = 1;
        plan.status = PlanStatus.DRAFT;
        plan.samplingType = SamplingType.RANDOM_SAMPLING.name();
        plan.aql = QcConstants.DEFAULT_AQL;
        plan.inspectionLevel = QcConstants.DEFAULT_INSPECTION_LEVEL;
        plan.sampleSize = QcConstants.DEFAULT_SAMPLE_SIZE;
        plan.acceptNumber = QcConstants.DEFAULT_ACCEPT_NUMBER;
        plan.rejectNumber = QcConstants.DEFAULT_REJECT_NUMBER;
        plan.dispositionRule = QcConstants.DEFAULT_DISPOSITION_RULE;
        plan.qualifiedFlow = QcConstants.DEFAULT_QUALIFIED_FLOW;
        plan.unqualifiedFlow = QcConstants.DEFAULT_UNQUALIFIED_FLOW;
        plan.sortOrder = QcConstants.DEFAULT_SORT_ORDER;
        plan.createdAt = LocalDateTime.now();
        plan.updatedAt = LocalDateTime.now();
        return plan;
    }
    
    public void update(String planName, String inspectionType, String applicableProductTypes,
            String samplingType, String aql, String inspectionLevel, Integer sampleSize,
            String acceptNumber, String rejectNumber, String dispositionRule,
            String qualifiedFlow, String unqualifiedFlow, String specialApprovalFlow,
            LocalDateTime effectiveDate, LocalDateTime expirationDate, String remark) {
        this.planName = planName;
        this.inspectionType = inspectionType;
        this.applicableProductTypes = applicableProductTypes;
        this.samplingType = samplingType;
        this.aql = aql;
        this.inspectionLevel = inspectionLevel;
        this.sampleSize = sampleSize;
        this.acceptNumber = acceptNumber;
        this.rejectNumber = rejectNumber;
        this.dispositionRule = dispositionRule;
        this.qualifiedFlow = qualifiedFlow;
        this.unqualifiedFlow = unqualifiedFlow;
        this.specialApprovalFlow = specialApprovalFlow;
        this.effectiveDate = effectiveDate;
        this.expirationDate = expirationDate;
        this.remark = remark;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = PlanStatus.EFFECTIVE;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void deactivate() {
        this.status = PlanStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void expire() {
        this.status = PlanStatus.EXPIRED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addPlanItem(QcPlanItem item) {
        this.planItems.add(item);
    }
    
    public void removePlanItem(Long itemId) {
        this.planItems.removeIf(item -> item.getId().equals(itemId));
    }
    
    public boolean isEffective() {
        return this.status == PlanStatus.EFFECTIVE;
    }
    
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
    public PlanStatus getStatus() { return status; }
    public void setStatus(PlanStatus status) { this.status = status; }
    public LocalDateTime getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
    public LocalDateTime getExpirationDate() { return expirationDate; }
    public void setExpirationDate(LocalDateTime expirationDate) { this.expirationDate = expirationDate; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public List<QcPlanItem> getPlanItems() { return planItems; }
    public void setPlanItems(List<QcPlanItem> planItems) { this.planItems = planItems; }
}