package com.metawebthree.finance.infrastructure.persistence.dataobject.asset;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class FixedAssetDisposalDO {
    private Long id;
    private String disposalCode;
    private String disposalType;
    private Long assetId;
    private String assetCode;
    private String assetName;
    private BigDecimal originalValue;
    private BigDecimal netValue;
    private BigDecimal accumulatedDepreciation;
    private BigDecimal disposalAmount;
    private String disposalDate;
    private String status;
    private String disposalReason;
    private String disposalMethod;
    private String acquirerName;
    private String acquirerContact;
    private BigDecimal gainLoss;
    private String voucherNumber;
    private String approvalStatus;
    private String approvalComment;
    private Long approverId;
    private String approverName;
    private String approvalDate;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public String getDisposalCode() { return disposalCode; }
    public String getDisposalType() { return disposalType; }
    public Long getAssetId() { return assetId; }
    public String getAssetCode() { return assetCode; }
    public String getAssetName() { return assetName; }
    public BigDecimal getOriginalValue() { return originalValue; }
    public BigDecimal getNetValue() { return netValue; }
    public BigDecimal getAccumulatedDepreciation() { return accumulatedDepreciation; }
    public BigDecimal getDisposalAmount() { return disposalAmount; }
    public String getDisposalDate() { return disposalDate; }
    public String getStatus() { return status; }
    public String getDisposalReason() { return disposalReason; }
    public String getDisposalMethod() { return disposalMethod; }
    public String getAcquirerName() { return acquirerName; }
    public String getAcquirerContact() { return acquirerContact; }
    public BigDecimal getGainLoss() { return gainLoss; }
    public String getVoucherNumber() { return voucherNumber; }
    public String getApprovalStatus() { return approvalStatus; }
    public String getApprovalComment() { return approvalComment; }
    public Long getApproverId() { return approverId; }
    public String getApproverName() { return approverName; }
    public String getApprovalDate() { return approvalDate; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setDisposalCode(String disposalCode) { this.disposalCode = disposalCode; }
    public void setDisposalType(String disposalType) { this.disposalType = disposalType; }
    public void setAssetId(Long assetId) { this.assetId = assetId; }
    public void setAssetCode(String assetCode) { this.assetCode = assetCode; }
    public void setAssetName(String assetName) { this.assetName = assetName; }
    public void setOriginalValue(BigDecimal originalValue) { this.originalValue = originalValue; }
    public void setNetValue(BigDecimal netValue) { this.netValue = netValue; }
    public void setAccumulatedDepreciation(BigDecimal accumulatedDepreciation) { this.accumulatedDepreciation = accumulatedDepreciation; }
    public void setDisposalAmount(BigDecimal disposalAmount) { this.disposalAmount = disposalAmount; }
    public void setDisposalDate(String disposalDate) { this.disposalDate = disposalDate; }
    public void setStatus(String status) { this.status = status; }
    public void setDisposalReason(String disposalReason) { this.disposalReason = disposalReason; }
    public void setDisposalMethod(String disposalMethod) { this.disposalMethod = disposalMethod; }
    public void setAcquirerName(String acquirerName) { this.acquirerName = acquirerName; }
    public void setAcquirerContact(String acquirerContact) { this.acquirerContact = acquirerContact; }
    public void setGainLoss(BigDecimal gainLoss) { this.gainLoss = gainLoss; }
    public void setVoucherNumber(String voucherNumber) { this.voucherNumber = voucherNumber; }
    public void setApprovalStatus(String approvalStatus) { this.approvalStatus = approvalStatus; }
    public void setApprovalComment(String approvalComment) { this.approvalComment = approvalComment; }
    public void setApproverId(Long approverId) { this.approverId = approverId; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setApprovalDate(String approvalDate) { this.approvalDate = approvalDate; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}