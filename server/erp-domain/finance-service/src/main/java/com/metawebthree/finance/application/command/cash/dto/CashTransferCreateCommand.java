package com.metawebthree.finance.application.command.cash.dto;

import java.math.BigDecimal;

public class CashTransferCreateCommand {
    private String transferNo;
    private Long fromAccountId;
    private String fromAccountName;
    private Long toAccountId;
    private String toAccountName;
    private BigDecimal amount;
    private String currency;
    private String type;
    private String purpose;
    private Long createdBy;
    private String creatorName;
    private String remark;

    public String getTransferNo() { return transferNo; }
    public void setTransferNo(String transferNo) { this.transferNo = transferNo; }
    public Long getFromAccountId() { return fromAccountId; }
    public void setFromAccountId(Long fromAccountId) { this.fromAccountId = fromAccountId; }
    public String getFromAccountName() { return fromAccountName; }
    public void setFromAccountName(String fromAccountName) { this.fromAccountName = fromAccountName; }
    public Long getToAccountId() { return toAccountId; }
    public void setToAccountId(Long toAccountId) { this.toAccountId = toAccountId; }
    public String getToAccountName() { return toAccountName; }
    public void setToAccountName(String toAccountName) { this.toAccountName = toAccountName; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getPurpose() { return purpose; }
    public void setPurpose(String purpose) { this.purpose = purpose; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}