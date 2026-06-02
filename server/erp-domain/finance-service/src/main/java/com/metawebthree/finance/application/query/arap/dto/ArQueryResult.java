package com.metawebthree.finance.application.query.arap.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class ArQueryResult {
    private Long id;
    private String arCode;
    private Long customerId;
    private String customerName;
    private String businessType;
    private String relatedDocumentType;
    private String relatedDocumentNo;
    private BigDecimal amount;
    private BigDecimal receivedAmount;
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
    public String getArCode() { return arCode; }
    public void setArCode(String arCode) { this.arCode = arCode; }
    public Long getCustomerId() { return customerId; }
    public void setCustomerId(Long customerId) { this.customerId = customerId; }
    public String getCustomerName() { return customerName; }
    public void setCustomerName(String customerName) { this.customerName = customerName; }
    public String getBusinessType() { return businessType; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public String getRelatedDocumentType() { return relatedDocumentType; }
    public void setRelatedDocumentType(String relatedDocumentType) { this.relatedDocumentType = relatedDocumentType; }
    public String getRelatedDocumentNo() { return relatedDocumentNo; }
    public void setRelatedDocumentNo(String relatedDocumentNo) { this.relatedDocumentNo = relatedDocumentNo; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public BigDecimal getReceivedAmount() { return receivedAmount; }
    public void setReceivedAmount(BigDecimal receivedAmount) { this.receivedAmount = receivedAmount; }
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

    public static ArQueryResult fromEntity(com.metawebthree.finance.domain.entity.arap.AccountsReceivable ar) {
        ArQueryResult result = new ArQueryResult();
        result.setId(ar.getId());
        result.setArCode(ar.getArCode());
        result.setCustomerId(ar.getCustomerId());
        result.setCustomerName(ar.getCustomerName());
        result.setBusinessType(ar.getBusinessType());
        result.setRelatedDocumentType(ar.getRelatedDocumentType());
        result.setRelatedDocumentNo(ar.getRelatedDocumentNo());
        result.setAmount(ar.getAmount());
        result.setReceivedAmount(ar.getReceivedAmount());
        result.setRemainingAmount(ar.getRemainingAmount());
        result.setInvoiceDate(ar.getInvoiceDate());
        result.setDueDate(ar.getDueDate());
        result.setCreditTerm(ar.getCreditTerm());
        result.setStatus(ar.getStatus() != null ? ar.getStatus().name() : null);
        result.setCurrency(ar.getCurrency());
        result.setExchangeRate(ar.getExchangeRate());
        result.setDescription(ar.getDescription());
        result.setCreatedBy(ar.getCreatedBy());
        result.setCreatorName(ar.getCreatorName());
        result.setCreatedAt(ar.getCreatedAt());
        result.setUpdatedAt(ar.getUpdatedAt());
        return result;
    }
}