package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class BankReconciliation {
    private Long id;
    private String reconciliationNo;
    private Long bankAccountId;
    private String bankAccountName;
    private String bankName;
    private LocalDate statementDate;
    private LocalDate statementEndDate;
    private BigDecimal bankBalance;
    private BigDecimal bookBalance;
    private BigDecimal variance;
    private ReconciliationStatus status;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime approvedAt;
    private Long approvedBy;
    private String approverName;
    private List<ReconciliationItem> items;

    public enum ReconciliationStatus {
        DRAFT, PENDING_APPROVAL, APPROVED, ADJUSTED
    }

    public void create(String reconciliationNo, Long bankAccountId, String bankAccountName,
                       String bankName, LocalDate statementDate, LocalDate statementEndDate,
                       BigDecimal bankBalance, BigDecimal bookBalance, Long createdBy, String creatorName) {
        this.reconciliationNo = reconciliationNo;
        this.bankAccountId = bankAccountId;
        this.bankAccountName = bankAccountName;
        this.bankName = bankName;
        this.statementDate = statementDate;
        this.statementEndDate = statementEndDate;
        this.bankBalance = bankBalance;
        this.bookBalance = bookBalance;
        this.variance = bankBalance.subtract(bookBalance);
        this.status = ReconciliationStatus.DRAFT;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.items = new ArrayList<>();
    }

    public void submitForApproval() {
        if (status != ReconciliationStatus.DRAFT) {
            return;
        }
        status = ReconciliationStatus.PENDING_APPROVAL;
        updatedAt = LocalDateTime.now();
    }

    public void approve(Long approvedBy, String approverName) {
        if (status != ReconciliationStatus.PENDING_APPROVAL) {
            return;
        }
        status = ReconciliationStatus.APPROVED;
        this.approvedBy = approvedBy;
        this.approverName = approverName;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addItem(ReconciliationItem item) {
        if (items == null) {
            items = new ArrayList<>();
        }
        items.add(item);
        updatedAt = LocalDateTime.now();
    }

    public void updateBalances(BigDecimal bankBalance, BigDecimal bookBalance) {
        this.bankBalance = bankBalance;
        this.bookBalance = bookBalance;
        this.variance = bankBalance.subtract(bookBalance);
        updatedAt = LocalDateTime.now();
    }

    public boolean isBalanced() {
        return variance.compareTo(BigDecimal.ZERO) == 0;
    }

    public BigDecimal getUnreconciledAmount() {
        if (items == null) {
            return variance.abs();
        }
        return items.stream()
                .filter(item -> item.getReconciliationStatus() == ReconciliationItem.ReconciliationItemStatus.PENDING)
                .map(ReconciliationItem::getAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getReconciliationNo() { return reconciliationNo; }
    public Long getBankAccountId() { return bankAccountId; }
    public String getBankAccountName() { return bankAccountName; }
    public String getBankName() { return bankName; }
    public LocalDate getStatementDate() { return statementDate; }
    public LocalDate getStatementEndDate() { return statementEndDate; }
    public BigDecimal getBankBalance() { return bankBalance; }
    public BigDecimal getBookBalance() { return bookBalance; }
    public BigDecimal getVariance() { return variance; }
    public ReconciliationStatus getStatus() { return status; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public Long getApprovedBy() { return approvedBy; }
    public String getApproverName() { return approverName; }
    public List<ReconciliationItem> getItems() { return items; }

    public void setId(Long id) { this.id = id; }
    public void setReconciliationNo(String reconciliationNo) { this.reconciliationNo = reconciliationNo; }
    public void setBankAccountId(Long bankAccountId) { this.bankAccountId = bankAccountId; }
    public void setBankAccountName(String bankAccountName) { this.bankAccountName = bankAccountName; }
    public void setBankName(String bankName) { this.bankName = bankName; }
    public void setStatementDate(LocalDate statementDate) { this.statementDate = statementDate; }
    public void setStatementEndDate(LocalDate statementEndDate) { this.statementEndDate = statementEndDate; }
    public void setBankBalance(BigDecimal bankBalance) { this.bankBalance = bankBalance; }
    public void setBookBalance(BigDecimal bookBalance) { this.bookBalance = bookBalance; }
    public void setVariance(BigDecimal variance) { this.variance = variance; }
    public void setStatus(ReconciliationStatus status) { this.status = status; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public void setApprovedBy(Long approvedBy) { this.approvedBy = approvedBy; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setItems(List<ReconciliationItem> items) { this.items = items; }

    public static class ReconciliationItem {
        private Long id;
        private Long reconciliationId;
        private String itemType;
        private String itemNo;
        private LocalDate itemDate;
        private BigDecimal bankAmount;
        private BigDecimal bookAmount;
        private BigDecimal amount;
        private ReconciliationItemStatus reconciliationStatus;
        private String description;
        private String remark;

        public enum ReconciliationItemStatus {
            MATCHED, PENDING, ADJUSTED
        }

        // Getters and Setters
        public Long getId() { return id; }
        public Long getReconciliationId() { return reconciliationId; }
        public String getItemType() { return itemType; }
        public String getItemNo() { return itemNo; }
        public LocalDate getItemDate() { return itemDate; }
        public BigDecimal getBankAmount() { return bankAmount; }
        public BigDecimal getBookAmount() { return bookAmount; }
        public BigDecimal getAmount() { return amount; }
        public ReconciliationItemStatus getReconciliationStatus() { return reconciliationStatus; }
        public String getDescription() { return description; }
        public String getRemark() { return remark; }

        public void setId(Long id) { this.id = id; }
        public void setReconciliationId(Long reconciliationId) { this.reconciliationId = reconciliationId; }
        public void setItemType(String itemType) { this.itemType = itemType; }
        public void setItemNo(String itemNo) { this.itemNo = itemNo; }
        public void setItemDate(LocalDate itemDate) { this.itemDate = itemDate; }
        public void setBankAmount(BigDecimal bankAmount) { this.bankAmount = bankAmount; }
        public void setBookAmount(BigDecimal bookAmount) { this.bookAmount = bookAmount; }
        public void setAmount(BigDecimal amount) { this.amount = amount; }
        public void setReconciliationStatus(ReconciliationItemStatus reconciliationStatus) { this.reconciliationStatus = reconciliationStatus; }
        public void setDescription(String description) { this.description = description; }
        public void setRemark(String remark) { this.remark = remark; }
    }
}