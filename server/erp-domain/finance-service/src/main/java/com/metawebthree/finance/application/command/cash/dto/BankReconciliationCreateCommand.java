package com.metawebthree.finance.application.command.cash.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class BankReconciliationCreateCommand {
    private String reconciliationNo;
    private Long bankAccountId;
    private String bankAccountName;
    private String bankName;
    private LocalDate statementDate;
    private LocalDate statementEndDate;
    private BigDecimal bankBalance;
    private BigDecimal bookBalance;
    private Long createdBy;
    private String creatorName;
    private String remark;

    public String getReconciliationNo() { return reconciliationNo; }
    public void setReconciliationNo(String reconciliationNo) { this.reconciliationNo = reconciliationNo; }
    public Long getBankAccountId() { return bankAccountId; }
    public void setBankAccountId(Long bankAccountId) { this.bankAccountId = bankAccountId; }
    public String getBankAccountName() { return bankAccountName; }
    public void setBankAccountName(String bankAccountName) { this.bankAccountName = bankAccountName; }
    public String getBankName() { return bankName; }
    public void setBankName(String bankName) { this.bankName = bankName; }
    public LocalDate getStatementDate() { return statementDate; }
    public void setStatementDate(LocalDate statementDate) { this.statementDate = statementDate; }
    public LocalDate getStatementEndDate() { return statementEndDate; }
    public void setStatementEndDate(LocalDate statementEndDate) { this.statementEndDate = statementEndDate; }
    public BigDecimal getBankBalance() { return bankBalance; }
    public void setBankBalance(BigDecimal bankBalance) { this.bankBalance = bankBalance; }
    public BigDecimal getBookBalance() { return bookBalance; }
    public void setBookBalance(BigDecimal bookBalance) { this.bookBalance = bookBalance; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}