package com.metawebthree.finance.application.command.arap.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class ApCreateCommand {
    private String apCode;
    private Long supplierId;
    private String supplierName;
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

    public String getApCode() { return apCode; }
    public void setApCode(String apCode) { this.apCode = apCode; }
    public Long getSupplierId() { return supplierId; }
    public void setSupplierId(Long supplierId) { this.supplierId = supplierId; }
    public String getSupplierName() { return supplierName; }
    public void setSupplierName(String supplierName) { this.supplierName = supplierName; }
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