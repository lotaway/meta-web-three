package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CashTransfer {
    private Long id;
    private String transferNo;
    private Long fromAccountId;
    private String fromAccountName;
    private Long toAccountId;
    private String toAccountName;
    private BigDecimal amount;
    private String currency;
    private CashTransferStatus status;
    private CashTransferType type;
    private String purpose;
    private LocalDateTime transferDate;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime approvedAt;
    private Long approvedBy;
    private String approverName;
    private LocalDateTime executedAt;
    private String executorName;

    public enum CashTransferStatus {
        DRAFT, PENDING_APPROVAL, APPROVED, REJECTED, EXECUTED, CANCELLED
    }

    public enum CashTransferType {
        INTERNA_TRANSFER, EXTERNAL_TRANSFER
    }

    public void create(String transferNo, Long fromAccountId, String fromAccountName,
                       Long toAccountId, String toAccountName, BigDecimal amount,
                       String currency, CashTransferType type, String purpose,
                       Long createdBy, String creatorName) {
        this.transferNo = transferNo;
        this.fromAccountId = fromAccountId;
        this.fromAccountName = fromAccountName;
        this.toAccountId = toAccountId;
        this.toAccountName = toAccountName;
        this.amount = amount;
        this.currency = currency;
        this.type = type;
        this.purpose = purpose;
        this.status = CashTransferStatus.DRAFT;
        this.transferDate = LocalDateTime.now();
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void submitForApproval() {
        if (status != CashTransferStatus.DRAFT) {
            return;
        }
        if (fromAccountId.equals(toAccountId)) {
            throw new IllegalStateException("Cannot transfer to the same account");
        }
        status = CashTransferStatus.PENDING_APPROVAL;
        updatedAt = LocalDateTime.now();
    }

    public void approve(Long approvedBy, String approverName) {
        if (status != CashTransferStatus.PENDING_APPROVAL) {
            return;
        }
        status = CashTransferStatus.APPROVED;
        this.approvedBy = approvedBy;
        this.approverName = approverName;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reject() {
        if (status != CashTransferStatus.PENDING_APPROVAL) {
            return;
        }
        status = CashTransferStatus.REJECTED;
        updatedAt = LocalDateTime.now();
    }

    public void execute(String executorName) {
        if (status != CashTransferStatus.APPROVED) {
            return;
        }
        status = CashTransferStatus.EXECUTED;
        this.executorName = executorName;
        this.executedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status != CashTransferStatus.DRAFT && status != CashTransferStatus.REJECTED) {
            return;
        }
        status = CashTransferStatus.CANCELLED;
        updatedAt = LocalDateTime.now();
    }

    public boolean isInternal() {
        return type == CashTransferType.INTERNA_TRANSFER;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getTransferNo() { return transferNo; }
    public Long getFromAccountId() { return fromAccountId; }
    public String getFromAccountName() { return fromAccountName; }
    public Long getToAccountId() { return toAccountId; }
    public String getToAccountName() { return toAccountName; }
    public BigDecimal getAmount() { return amount; }
    public String getCurrency() { return currency; }
    public CashTransferStatus getStatus() { return status; }
    public CashTransferType getType() { return type; }
    public String getPurpose() { return purpose; }
    public LocalDateTime getTransferDate() { return transferDate; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public Long getApprovedBy() { return approvedBy; }
    public String getApproverName() { return approverName; }
    public LocalDateTime getExecutedAt() { return executedAt; }
    public String getExecutorName() { return executorName; }

    public void setId(Long id) { this.id = id; }
    public void setTransferNo(String transferNo) { this.transferNo = transferNo; }
    public void setFromAccountId(Long fromAccountId) { this.fromAccountId = fromAccountId; }
    public void setFromAccountName(String fromAccountName) { this.fromAccountName = fromAccountName; }
    public void setToAccountId(Long toAccountId) { this.toAccountId = toAccountId; }
    public void setToAccountName(String toAccountName) { this.toAccountName = toAccountName; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setStatus(CashTransferStatus status) { this.status = status; }
    public void setType(CashTransferType type) { this.type = type; }
    public void setPurpose(String purpose) { this.purpose = purpose; }
    public void setTransferDate(LocalDateTime transferDate) { this.transferDate = transferDate; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public void setApprovedBy(Long approvedBy) { this.approvedBy = approvedBy; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setExecutedAt(LocalDateTime executedAt) { this.executedAt = executedAt; }
    public void setExecutorName(String executorName) { this.executorName = executorName; }
}