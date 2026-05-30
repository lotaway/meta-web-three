package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class BankAccount {
    private Long id;
    private String accountCode;
    private String accountName;
    private String bankName;
    private String accountNumber;
    private String accountType;
    private BankAccountStatus status;
    private BigDecimal balance;
    private BigDecimal frozenAmount;
    private String currency;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Boolean isActive;

    public enum BankAccountStatus {
        ACTIVE, FROZEN, CLOSED
    }

    public void create(String accountCode, String accountName, String bankName,
                       String accountNumber, String accountType, String currency,
                       Long createdBy, String creatorName) {
        this.accountCode = accountCode;
        this.accountName = accountName;
        this.bankName = bankName;
        this.accountNumber = accountNumber;
        this.accountType = accountType;
        this.status = BankAccountStatus.ACTIVE;
        this.balance = BigDecimal.ZERO;
        this.frozenAmount = BigDecimal.ZERO;
        this.currency = currency;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.isActive = true;
    }

    public void freeze() {
        if (status == BankAccountStatus.ACTIVE) {
            status = BankAccountStatus.FROZEN;
            updatedAt = LocalDateTime.now();
        }
    }

    public void unfreeze() {
        if (status == BankAccountStatus.FROZEN) {
            status = BankAccountStatus.ACTIVE;
            updatedAt = LocalDateTime.now();
        }
    }

    public void close() {
        status = BankAccountStatus.CLOSED;
        isActive = false;
        updatedAt = LocalDateTime.now();
    }

    public void deposit(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            return;
        }
        balance = balance.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public void withdraw(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            return;
        }
        BigDecimal available = balance.subtract(frozenAmount);
        if (available.compareTo(amount) < 0) {
            throw new IllegalStateException("Insufficient balance");
        }
        balance = balance.subtract(amount);
        updatedAt = LocalDateTime.now();
    }

    public void freezeAmount(BigDecimal amount) {
        BigDecimal available = balance.subtract(frozenAmount);
        if (available.compareTo(amount) < 0) {
            throw new IllegalStateException("Insufficient available balance");
        }
        frozenAmount = frozenAmount.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public void unfreezeAmount(BigDecimal amount) {
        if (frozenAmount.compareTo(amount) < 0) {
            throw new IllegalStateException("Invalid unfreeze amount");
        }
        frozenAmount = frozenAmount.subtract(amount);
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getAvailableBalance() {
        return balance.subtract(frozenAmount);
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getAccountCode() { return accountCode; }
    public String getAccountName() { return accountName; }
    public String getBankName() { return bankName; }
    public String getAccountNumber() { return accountNumber; }
    public String getAccountType() { return accountType; }
    public BankAccountStatus getStatus() { return status; }
    public BigDecimal getBalance() { return balance; }
    public BigDecimal getFrozenAmount() { return frozenAmount; }
    public String getCurrency() { return currency; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public Boolean getIsActive() { return isActive; }

    public void setId(Long id) { this.id = id; }
    public void setAccountCode(String accountCode) { this.accountCode = accountCode; }
    public void setAccountName(String accountName) { this.accountName = accountName; }
    public void setBankName(String bankName) { this.bankName = bankName; }
    public void setAccountNumber(String accountNumber) { this.accountNumber = accountNumber; }
    public void setAccountType(String accountType) { this.accountType = accountType; }
    public void setStatus(BankAccountStatus status) { this.status = status; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
    public void setFrozenAmount(BigDecimal frozenAmount) { this.frozenAmount = frozenAmount; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
}