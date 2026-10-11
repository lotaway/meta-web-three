package com.metawebthree.invoice.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class Invoice {
    private Long id;
    private String invoiceNo;
    private String orderNo;
    private Long customerId;
    private String customerName;
    private String customerTaxNo;
    private String customerAddress;
    private String customerBank;
    private String customerAccount;
    private InvoiceType type;
    private InvoiceStatus status;
    private BigDecimal amount;
    private BigDecimal taxAmount;
    private BigDecimal totalAmount;
    private String taxRate;
    private LocalDateTime issueDate;
    private String issuer;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum InvoiceType {
        VAT_SPECIAL, VAT_NORMAL, ELECTRONIC, RECEIPT
    }

    public enum InvoiceStatus {
        DRAFT, PENDING, ISSUED, PRINTED, VOIDED, RED_FLUSHED
    }

    public void createDraft(String invoiceNo, String orderNo, Long customerId, String customerName,
                            String customerTaxNo, InvoiceType type, BigDecimal amount, String taxRate) {
        this.invoiceNo = invoiceNo;
        this.orderNo = orderNo;
        this.customerId = customerId;
        this.customerName = customerName;
        this.customerTaxNo = customerTaxNo;
        this.type = type;
        this.amount = amount;
        this.taxRate = taxRate;
        this.taxAmount = calculateTax(amount, taxRate);
        this.totalAmount = amount.add(this.taxAmount);
        this.status = InvoiceStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    private BigDecimal calculateTax(BigDecimal amount, String taxRate) {
        BigDecimal rate = new BigDecimal(taxRate.replace("%", ""));
        return amount.multiply(rate).divide(BigDecimal.valueOf(100));
    }

    public void issue(String issuer) {
        if (status != InvoiceStatus.DRAFT && status != InvoiceStatus.PENDING) {
            throw new IllegalStateException("Cannot issue invoice in current status");
        }
        this.status = InvoiceStatus.ISSUED;
        this.issueDate = LocalDateTime.now();
        this.issuer = issuer;
        this.updatedAt = LocalDateTime.now();
    }

    public void print() {
        if (status != InvoiceStatus.ISSUED) {
            throw new IllegalStateException("Cannot print non-issued invoice");
        }
        this.status = InvoiceStatus.PRINTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void voidInvoice(String reason) {
        if (status == InvoiceStatus.VOIDED || status == InvoiceStatus.RED_FLUSHED) {
            throw new IllegalStateException("Invoice already voided or red-flushed");
        }
        this.status = InvoiceStatus.VOIDED;
        this.remark = reason;
        this.updatedAt = LocalDateTime.now();
    }

    public void redFlush(String reason) {
        if (status != InvoiceStatus.ISSUED && status != InvoiceStatus.PRINTED) {
            throw new IllegalStateException("Cannot red-flush non-issued invoice");
        }
        if (totalAmount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalStateException("Invoice amount must be positive for red-flush");
        }
        this.status = InvoiceStatus.RED_FLUSHED;
        this.remark = "RED_FLUSH: " + reason;
        this.totalAmount = totalAmount.negate();
        this.taxAmount = taxAmount.negate();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateCustomerInfo(String name, String taxNo, String address, String bank, String account) {
        this.customerName = name;
        this.customerTaxNo = taxNo;
        this.customerAddress = address;
        this.customerBank = bank;
        this.customerAccount = account;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getInvoiceNo() { return invoiceNo; }
    public String getOrderNo() { return orderNo; }
    public Long getCustomerId() { return customerId; }
    public String getCustomerName() { return customerName; }
    public String getCustomerTaxNo() { return customerTaxNo; }
    public String getCustomerAddress() { return customerAddress; }
    public String getCustomerBank() { return customerBank; }
    public String getCustomerAccount() { return customerAccount; }
    public InvoiceType getType() { return type; }
    public InvoiceStatus getStatus() { return status; }
    public BigDecimal getAmount() { return amount; }
    public BigDecimal getTaxAmount() { return taxAmount; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public String getTaxRate() { return taxRate; }
    public LocalDateTime getIssueDate() { return issueDate; }
    public String getIssuer() { return issuer; }
    public String getRemark() { return remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
    public void setInvoiceNo(String invoiceNo) { this.invoiceNo = invoiceNo; }
    public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
    public void setCustomerId(Long customerId) { this.customerId = customerId; }
    public void setCustomerName(String customerName) { this.customerName = customerName; }
    public void setCustomerTaxNo(String customerTaxNo) { this.customerTaxNo = customerTaxNo; }
    public void setCustomerAddress(String customerAddress) { this.customerAddress = customerAddress; }
    public void setCustomerBank(String customerBank) { this.customerBank = customerBank; }
    public void setCustomerAccount(String customerAccount) { this.customerAccount = customerAccount; }
    public void setType(InvoiceType type) { this.type = type; }
    public void setStatus(InvoiceStatus status) { this.status = status; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public void setTaxAmount(BigDecimal taxAmount) { this.taxAmount = taxAmount; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public void setTaxRate(String taxRate) { this.taxRate = taxRate; }
    public void setIssueDate(LocalDateTime issueDate) { this.issueDate = issueDate; }
    public void setIssuer(String issuer) { this.issuer = issuer; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}