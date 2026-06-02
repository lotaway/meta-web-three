package com.metawebthree.finance.domain.entity.arap;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class AccountsReceivable {
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
    private ArStatus status;
    private String currency;
    private BigDecimal exchangeRate;
    private BigDecimal originalAmount;
    private String description;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Boolean isActive;

    public enum ArStatus {
        PENDING, PARTIAL_RECEIVED, RECEIVED, OVERDUE, WRITE_OFF
    }

    public void create(String arCode, Long customerId, String customerName,
                       String businessType, BigDecimal amount, LocalDate invoiceDate,
                       Integer creditTerm, String currency, Long createdBy, String creatorName) {
        this.arCode = arCode;
        this.customerId = customerId;
        this.customerName = customerName;
        this.businessType = businessType;
        this.amount = amount;
        this.receivedAmount = BigDecimal.ZERO;
        this.remainingAmount = amount;
        this.invoiceDate = invoiceDate;
        this.creditTerm = creditTerm != null ? creditTerm : 30;
        this.dueDate = invoiceDate.plusDays(this.creditTerm);
        this.status = ArStatus.PENDING;
        this.currency = currency;
        this.exchangeRate = BigDecimal.ONE;
        this.originalAmount = amount;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.isActive = true;
    }

    public void receive(BigDecimal receiveAmount) {
        if (receiveAmount.compareTo(BigDecimal.ZERO) <= 0) {
            return;
        }
        BigDecimal canReceive = remainingAmount;
        if (receiveAmount.compareTo(canReceive) > 0) {
            receiveAmount = canReceive;
        }
        receivedAmount = receivedAmount.add(receiveAmount);
        remainingAmount = remainingAmount.subtract(receiveAmount);
        
        if (remainingAmount.compareTo(BigDecimal.ZERO) == 0) {
            status = ArStatus.RECEIVED;
        } else {
            status = ArStatus.PARTIAL_RECEIVED;
        }
        updatedAt = LocalDateTime.now();
    }

    public void checkOverdue() {
        if (status == ArStatus.PENDING || status == ArStatus.PARTIAL_RECEIVED) {
            if (LocalDate.now().isAfter(dueDate)) {
                status = ArStatus.OVERDUE;
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
        receivedAmount = receivedAmount.add(writeOffAmount);
        remainingAmount = remainingAmount.subtract(writeOffAmount);
        
        if (remainingAmount.compareTo(BigDecimal.ZERO) == 0) {
            status = ArStatus.WRITE_OFF;
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
        this.remainingAmount = this.amount.subtract(receivedAmount);
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getArCode() { return arCode; }
    public Long getCustomerId() { return customerId; }
    public String getCustomerName() { return customerName; }
    public String getBusinessType() { return businessType; }
    public String getRelatedDocumentType() { return relatedDocumentType; }
    public String getRelatedDocumentNo() { return relatedDocumentNo; }
    public BigDecimal getAmount() { return amount; }
    public BigDecimal getReceivedAmount() { return receivedAmount; }
    public BigDecimal getRemainingAmount() { return remainingAmount; }
    public LocalDate getInvoiceDate() { return invoiceDate; }
    public LocalDate getDueDate() { return dueDate; }
    public Integer getCreditTerm() { return creditTerm; }
    public ArStatus getStatus() { return status; }
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
    public void setArCode(String arCode) { this.arCode = arCode; }
    public void setCustomerId(Long customerId) { this.customerId = customerId; }
    public void setCustomerName(String customerName) { this.customerName = customerName; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public void setRelatedDocumentType(String relatedDocumentType) { this.relatedDocumentType = relatedDocumentType; }
    public void setRelatedDocumentNo(String relatedDocumentNo) { this.relatedDocumentNo = relatedDocumentNo; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public void setReceivedAmount(BigDecimal receivedAmount) { this.receivedAmount = receivedAmount; }
    public void setRemainingAmount(BigDecimal remainingAmount) { this.remainingAmount = remainingAmount; }
    public void setInvoiceDate(LocalDate invoiceDate) { this.invoiceDate = invoiceDate; }
    public void setDueDate(LocalDate dueDate) { this.dueDate = dueDate; }
    public void setCreditTerm(Integer creditTerm) { this.creditTerm = creditTerm; }
    public void setStatus(ArStatus status) { this.status = status; }
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