package com.metawebthree.finance.application.command.asset.dto;

import java.math.BigDecimal;

public class AssetDisposalCreateCommand {
    private String disposalCode;
    private String disposalType;
    private Long assetId;
    private String disposalDate;
    private String disposalReason;
    private String disposalMethod;
    private String acquirerName;
    private String acquirerContact;
    private BigDecimal disposalAmount;
    private String remark;
    private Long createdBy;
    private String creatorName;

    public String getDisposalCode() { return disposalCode; }
    public String getDisposalType() { return disposalType; }
    public Long getAssetId() { return assetId; }
    public String getDisposalDate() { return disposalDate; }
    public String getDisposalReason() { return disposalReason; }
    public String getDisposalMethod() { return disposalMethod; }
    public String getAcquirerName() { return acquirerName; }
    public String getAcquirerContact() { return acquirerContact; }
    public BigDecimal getDisposalAmount() { return disposalAmount; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }

    public void setDisposalCode(String disposalCode) { this.disposalCode = disposalCode; }
    public void setDisposalType(String disposalType) { this.disposalType = disposalType; }
    public void setAssetId(Long assetId) { this.assetId = assetId; }
    public void setDisposalDate(String disposalDate) { this.disposalDate = disposalDate; }
    public void setDisposalReason(String disposalReason) { this.disposalReason = disposalReason; }
    public void setDisposalMethod(String disposalMethod) { this.disposalMethod = disposalMethod; }
    public void setAcquirerName(String acquirerName) { this.acquirerName = acquirerName; }
    public void setAcquirerContact(String acquirerContact) { this.acquirerContact = acquirerContact; }
    public void setDisposalAmount(BigDecimal disposalAmount) { this.disposalAmount = disposalAmount; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
}