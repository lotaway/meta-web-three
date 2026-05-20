package com.metawebthree.settlement.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class SettlementOrder {
    private Long id;
    private String settlementNo;
    private String orderNo;
    private Long merchantId;
    private String merchantName;
    private BigDecimal orderAmount;
    private BigDecimal settlementAmount;
    private BigDecimal commissionAmount;
    private BigDecimal refundAmount;
    private SettlementStatus status;
    private String channel;
    private LocalDateTime settlementDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum SettlementStatus {
        PENDING, CONFIRMED, PROCESSING, COMPLETED, FAILED, CANCELLED
    }

    public void create(String settlementNo, String orderNo, Long merchantId, String merchantName, 
                       BigDecimal orderAmount, BigDecimal commissionRate) {
        this.settlementNo = settlementNo;
        this.orderNo = orderNo;
        this.merchantId = merchantId;
        this.merchantName = merchantName;
        this.orderAmount = orderAmount;
        this.commissionAmount = orderAmount.multiply(commissionRate);
        this.settlementAmount = orderAmount.subtract(commissionAmount);
        this.refundAmount = BigDecimal.ZERO;
        this.status = SettlementStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void confirm() {
        if (status != SettlementStatus.PENDING) {
            throw new IllegalStateException("Can only confirm pending settlements");
        }
        status = SettlementStatus.CONFIRMED;
        updatedAt = LocalDateTime.now();
    }

    public void process() {
        if (status != SettlementStatus.CONFIRMED) {
            throw new IllegalStateException("Can only process confirmed settlements");
        }
        status = SettlementStatus.PROCESSING;
        updatedAt = LocalDateTime.now();
    }

    public void complete() {
        if (status != SettlementStatus.PROCESSING) {
            throw new IllegalStateException("Can only complete processing settlements");
        }
        status = SettlementStatus.COMPLETED;
        this.settlementDate = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    public void fail(String reason) {
        if (status != SettlementStatus.PROCESSING) {
            throw new IllegalStateException("Can only fail processing settlements");
        }
        status = SettlementStatus.FAILED;
        this.description = reason;
        updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status != SettlementStatus.PENDING && status != SettlementStatus.CONFIRMED) {
            throw new IllegalStateException("Cannot cancel settlements in current status");
        }
        status = SettlementStatus.CANCELLED;
        updatedAt = LocalDateTime.now();
    }

    public void applyRefund(BigDecimal amount) {
        if (status == SettlementStatus.COMPLETED || status == SettlementStatus.CANCELLED) {
            throw new IllegalStateException("Cannot apply refund to settled settlements");
        }
        refundAmount = refundAmount.add(amount);
        settlementAmount = settlementAmount.subtract(amount);
        updatedAt = LocalDateTime.now();
    }

    private String description;

    public Long getId() { return id; }
    public String getSettlementNo() { return settlementNo; }
    public String getOrderNo() { return orderNo; }
    public Long getMerchantId() { return merchantId; }
    public String getMerchantName() { return merchantName; }
    public BigDecimal getOrderAmount() { return orderAmount; }
    public BigDecimal getSettlementAmount() { return settlementAmount; }
    public BigDecimal getCommissionAmount() { return commissionAmount; }
    public BigDecimal getRefundAmount() { return refundAmount; }
    public SettlementStatus getStatus() { return status; }
    public String getChannel() { return channel; }
    public LocalDateTime getSettlementDate() { return settlementDate; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
    public void setChannel(String channel) { this.channel = channel; }
}