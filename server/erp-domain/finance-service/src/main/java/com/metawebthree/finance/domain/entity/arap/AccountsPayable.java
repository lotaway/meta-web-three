package com.metawebthree.finance.domain.entity.arap;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class AccountsPayable {
    private Long id;
    private String apCode;
    private Long supplierId;
    private String supplierName;
    private String businessType;
    private String relatedDocumentType;
    private String relatedDocumentNo;
    private BigDecimal amount;
    private BigDecimal paidAmount;
    private BigDecimal remainingAmount;
    private LocalDate invoiceDate;
    private LocalDate dueDate;
    private Integer creditTerm;
    private ApStatus status;
    private String currency;
    private BigDecimal exchangeRate;
    private BigDecimal originalAmount;
    private String description;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Boolean isActive;

    public enum ApStatus {
        PENDING, PARTIAL_PAID, PAID, OVERDUE, WRITE_OFF
    }

    public void create(String apCode, Long supplierId, String supplierName,
                       String businessType, BigDecimal amount, LocalDate invoiceDate,
                       Integer creditTerm, String currency, Long createdBy, String creatorName) {
        this.apCode = apCode;
        this.supplierId = supplierId;
        this.supplierName = supplierName;
        this.businessType = businessType;
        this.amount = amount;
        this.paidAmount = BigDecimal.ZERO;
        this.remainingAmount = amount;
        this.invoiceDate = invoiceDate;
        this.creditTerm = creditTerm != null ? creditTerm : 30;
        this.dueDate = invoiceDate.plusDays(this.creditTerm);
        this.status = ApStatus.PENDING;
        this.currency = currency;
        this.exchangeRate = BigDecimal.ONE;
        this.originalAmount = amount;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.isActive = true;
    }

    public void pay(BigDecimal payAmount) {
        if (payAmount.compareTo(BigDecimal.ZERO) <= 0) {
            return;
        }
        BigDecimal canPay = remainingAmount;
        if (payAmount.compareTo(canPay) > 0) {
            payAmount = canPay;
        }
        paidAmount = paidAmount.add(payAmount);
        remainingAmount = remainingAmount.subtract(payAmount);
        
        if (remainingAmount.compareTo(BigDecimal.ZERO) == 0) {
            status = ApStatus.PAID;
        } else {
            status = ApStatus.PARTIAL_PAID;
        }
        updatedAt = LocalDateTime.now();
    }

    public void checkOverdue() {
        if (status == ApStatus.PENDING || status == ApStatus.PARTIAL_PAID) {
            if (LocalDate.now().isAfter(dueDate)) {
                status = ApStatus.OVERDUE;
                updatedAt = LocalDateTime.now();
            }
        }
    }

    public void writeOff(BigDecimal writeOffAmount) {
        if (writeOffAmount.compareTo(BigDecimal.ZERO) <= 0) {
            return;
        }
        if (writeOffAmount.compareTo(remainingAmount) > 0) {
            writeOffAmount = remainingAmount;
        }
        paidAmount = paidAmount.add(writeOffAmount);
        remainingAmount = remainingAmount.subtract(writeOffAmount);
        
        if (remainingAmount.compareTo(BigDecimal.ZERO) == 0) {
            status = ApStatus.WRITE_OFF;
        }
        updatedAt = LocalDateTime.now();
    }

    public void updateRelatedDocument(String documentType, String documentNo) {
        this.relatedDocumentType = documentType;
        this.relatedDocumentNo = documentNo;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateExchangeRate(BigDecimal newRate) {
        this.exchangeRate = newRate;
        this.amount = originalAmount.multiply(newRate);
        this.remainingAmount = this.amount.subtract(paidAmount);
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getApCode() { return apCode; }
    public Long getSupplierId() { return supplierId; }
    public String getSupplierName() { return supplierName; }
    public String getBusinessType() { return businessType; }
    public String getRelatedDocumentType() { return relatedDocumentType; }
    public String getRelatedDocumentNo() { return relatedDocumentNo; }
    public BigDecimal getAmount() { return amount; }
    public BigDecimal getPaidAmount() { return paidAmount; }
    public BigDecimal getRemainingAmount() { return remainingAmount; }
    public LocalDate getInvoiceDate() { return invoiceDate; }
    public LocalDate getDueDate() { return dueDate; }
    public Integer getCreditTerm() { return creditTerm; }
    public ApStatus getStatus() { return status; }
    public String getCurrency() { return currency; }
    public BigDecimal getExchangeRate() { return exchangeRate; }
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public String getDescription() { return description; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public Boolean getIsActive() { return isActive; }

    public void setId(Long id) { this.id = id; }
    public void setApCode(String apCode) { this.apCode = apCode; }
    public void setSupplierId(Long supplierId) { this.supplierId = supplierId; }
    public void setSupplierName(String supplierName) { this.supplierName = supplierName; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public void setRelatedDocumentType(String relatedDocumentType) { this.relatedDocumentType = relatedDocumentType; }
    public void setRelatedDocumentNo(String relatedDocumentNo) { this.relatedDocumentNo = relatedDocumentNo; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public void setPaidAmount(BigDecimal paidAmount) { this.paidAmount = paidAmount; }
    public void setRemainingAmount(BigDecimal remainingAmount) { this.remainingAmount = remainingAmount; }
    public void setInvoiceDate(LocalDate invoiceDate) { this.invoiceDate = invoiceDate; }
    public void setDueDate(LocalDate dueDate) { this.dueDate = dueDate; }
    public void setCreditTerm(Integer creditTerm) { this.creditTerm = creditTerm; }
    public void setStatus(ApStatus status) { this.status = status; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setExchangeRate(BigDecimal exchangeRate) { this.exchangeRate = exchangeRate; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
    public void setDescription(String description) { this.description = description; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
}