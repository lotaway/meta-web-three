package com.metawebthree.finance.application.command.arap.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class ArCreateCommand {
    private String arCode;
    private Long customerId;
    private String customerName;
    private String businessType;
    private BigDecimal amount;
    private LocalDate invoiceDate;
    private Integer creditTerm;
    private String currency;
    private String relatedDocumentType;
    private String relatedDocumentNo;
    private String description;
    private Long createdBy;
    private String creatorName;

    public String getArCode() { return arCode; }
    public void setArCode(String arCode) { this.arCode = arCode; }
    public Long getCustomerId() { return customerId; }
    public void setCustomerId(Long customerId) { this.customerId = customerId; }
    public String getCustomerName() { return customerName; }
    public void setCustomerName(String customerName) { this.customerName = customerName; }
    public String getBusinessType() { return businessType; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public LocalDate getInvoiceDate() { return invoiceDate; }
    public void setInvoiceDate(LocalDate invoiceDate) { this.invoiceDate = invoiceDate; }
    public Integer getCreditTerm() { return creditTerm; }
    public void setCreditTerm(Integer creditTerm) { this.creditTerm = creditTerm; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getRelatedDocumentType() { return relatedDocumentType; }
    public void setRelatedDocumentType(String relatedDocumentType) { this.relatedDocumentType = relatedDocumentType; }
    public String getRelatedDocumentNo() { return relatedDocumentNo; }
    public void setRelatedDocumentNo(String relatedDocumentNo) { this.relatedDocumentNo = relatedDocumentNo; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
}