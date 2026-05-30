package com.metawebthree.finance.domain.entity.budget;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class BudgetAdjustment {
    private Long id;
    private Long budgetId;
    private String budgetCode;
    private String adjustmentNo;
    private AdjustmentType type;
    private AdjustmentStatus status;
    private String subjectCode;
    private String subjectName;
    private BigDecimal originalAmount;
    private BigDecimal adjustedAmount;
    private BigDecimal afterAmount;
    private String reason;
    private Long applicantId;
    private String applicantName;
    private LocalDateTime appliedAt;
    private Long approverId;
    private String approverName;
    private LocalDateTime approvedAt;
    private String approvalComment;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum AdjustmentType {
        INCREASE, DECREASE, TRANSFER
    }

    public enum AdjustmentStatus {
        PENDING, APPROVED, REJECTED
    }

    public void apply(Long budgetId, String budgetCode, AdjustmentType type,
                      String subjectCode, String subjectName,
                      BigDecimal originalAmount, BigDecimal adjustedAmount,
                      Long applicantId, String applicantName, String reason) {
        this.budgetId = budgetId;
        this.budgetCode = budgetCode;
        this.type = type;
        this.subjectCode = subjectCode;
        this.subjectName = subjectName;
        this.originalAmount = originalAmount;
        this.adjustedAmount = adjustedAmount;
        this.afterAmount = originalAmount.add(adjustedAmount);
        this.reason = reason;
        this.applicantId = applicantId;
        this.applicantName = applicantName;
        this.status = AdjustmentStatus.PENDING;
        this.appliedAt = LocalDateTime.now();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.adjustmentNo = generateAdjustmentNo();
    }

    private String generateAdjustmentNo() {
        return "ADJ-" + System.currentTimeMillis();
    }

    public void approve(Long approverId, String approverName, String comment) {
        if (status != AdjustmentStatus.PENDING) {
            return;
        }
        this.approverId = approverId;
        this.approverName = approverName;
        this.approvalComment = comment;
        this.status = AdjustmentStatus.APPROVED;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reject(Long approverId, String approverName, String comment) {
        if (status != AdjustmentStatus.PENDING) {
            return;
        }
        this.approverId = approverId;
        this.approverName = approverName;
        this.approvalComment = comment;
        this.status = AdjustmentStatus.REJECTED;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public Long getBudgetId() { return budgetId; }
    public String getBudgetCode() { return budgetCode; }
    public String getAdjustmentNo() { return adjustmentNo; }
    public AdjustmentType getType() { return type; }
    public AdjustmentStatus getStatus() { return status; }
    public String getSubjectCode() { return subjectCode; }
    public String getSubjectName() { return subjectName; }
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public BigDecimal getAdjustedAmount() { return adjustedAmount; }
    public BigDecimal getAfterAmount() { return afterAmount; }
    public String getReason() { return reason; }
    public Long getApplicantId() { return applicantId; }
    public String getApplicantName() { return applicantName; }
    public LocalDateTime getAppliedAt() { return appliedAt; }
    public Long getApproverId() { return approverId; }
    public String getApproverName() { return approverName; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public String getApprovalComment() { return approvalComment; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setBudgetId(Long budgetId) { this.budgetId = budgetId; }
    public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
    public void setAdjustmentNo(String adjustmentNo) { this.adjustmentNo = adjustmentNo; }
    public void setType(AdjustmentType type) { this.type = type; }
    public void setStatus(AdjustmentStatus status) { this.status = status; }
    public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
    public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
    public void setAdjustedAmount(BigDecimal adjustedAmount) { this.adjustedAmount = adjustedAmount; }
    public void setAfterAmount(BigDecimal afterAmount) { this.afterAmount = afterAmount; }
    public void setReason(String reason) { this.reason = reason; }
    public void setApplicantId(Long applicantId) { this.applicantId = applicantId; }
    public void setApplicantName(String applicantName) { this.applicantName = applicantName; }
    public void setAppliedAt(LocalDateTime appliedAt) { this.appliedAt = appliedAt; }
    public void setApproverId(Long approverId) { this.approverId = approverId; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public void setApprovalComment(String approvalComment) { this.approvalComment = approvalComment; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}