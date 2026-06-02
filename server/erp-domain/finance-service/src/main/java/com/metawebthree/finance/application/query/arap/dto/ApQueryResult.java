package com.metawebthree.finance.application.query.arap.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class ApQueryResult {
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
    private String status;
    private String currency;
    private BigDecimal exchangeRate;
    private String description;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getApCode() { return apCode; }
    public void setApCode(String apCode) { this.apCode = apCode; }
    public Long getSupplierId() { return supplierId; }
    public void setSupplierId(Long supplierId) { this.supplierId = supplierId; }
    public String getSupplierName() { return supplierName; }
    public void setSupplierName(String supplierName) { this.supplierName = supplierName; }
    public String getBusinessType() { return businessType; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public String getRelatedDocumentType() { return relatedDocumentType; }
    public void setRelatedDocumentType(String relatedDocumentType) { this.relatedDocumentType = relatedDocumentType; }
    public String getRelatedDocumentNo() { return relatedDocumentNo; }
    public void setRelatedDocumentNo(String relatedDocumentNo) { this.relatedDocumentNo = relatedDocumentNo; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public BigDecimal getPaidAmount() { return paidAmount; }
    public void setPaidAmount(BigDecimal paidAmount) { this.paidAmount = paidAmount; }
    public BigDecimal getRemainingAmount() { return remainingAmount; }
    public void setRemainingAmount(BigDecimal remainingAmount) { this.remainingAmount = remainingAmount; }
    public LocalDate getInvoiceDate() { return invoiceDate; }
    public void setInvoiceDate(LocalDate invoiceDate) { this.invoiceDate = invoiceDate; }
    public LocalDate getDueDate() { return dueDate; }
    public void setDueDate(LocalDate dueDate) { this.dueDate = dueDate; }
    public Integer getCreditTerm() { return creditTerm; }
    public void setCreditTerm(Integer creditTerm) { this.creditTerm = creditTerm; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public BigDecimal getExchangeRate() { return exchangeRate; }
    public void setExchangeRate(BigDecimal exchangeRate) { this.exchangeRate = exchangeRate; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    public static ApQueryResult fromEntity(com.metawebthree.finance.domain.entity.arap.AccountsPayable ap) {
        ApQueryResult result = new ApQueryResult();
        result.setId(ap.getId());
        result.setApCode(ap.getApCode());
        result.setSupplierId(ap.getSupplierId());
        result.setSupplierName(ap.getSupplierName());
        result.setBusinessType(ap.getBusinessType());
        result.setRelatedDocumentType(ap.getRelatedDocumentType());
        result.setRelatedDocumentNo(ap.getRelatedDocumentNo());
        result.setAmount(ap.getAmount());
        result.setPaidAmount(ap.getPaidAmount());
        result.setRemainingAmount(ap.getRemainingAmount());
        result.setInvoiceDate(ap.getInvoiceDate());
        result.setDueDate(ap.getDueDate());
        result.setCreditTerm(ap.getCreditTerm());
        result.setStatus(ap.getStatus() != null ? ap.getStatus().name() : null);
        result.setCurrency(ap.getCurrency());
        result.setExchangeRate(ap.getExchangeRate());
        result.setDescription(ap.getDescription());
        result.setCreatedBy(ap.getCreatedBy());
        result.setCreatorName(ap.getCreatorName());
        result.setCreatedAt(ap.getCreatedAt());
        result.setUpdatedAt(ap.getUpdatedAt());
        return result;
    }
}