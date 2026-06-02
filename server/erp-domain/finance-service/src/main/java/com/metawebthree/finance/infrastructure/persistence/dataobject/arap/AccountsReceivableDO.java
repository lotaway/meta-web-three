package com.metawebthree.finance.infrastructure.persistence.dataobject.arap;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@TableName("accounts_receivable")
public class AccountsReceivableDO {
    @TableId(type = IdType.AUTO)
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
    private BigDecimal originalAmount;
    private String description;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Boolean isActive;

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
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
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
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
}