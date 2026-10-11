package com.metawebthree.settlement.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 物流运费结算单
 * 当物流订单完成配送时，自动生成运费结算单
 */
public class LogisticsSettlement {
    private Long id;
    private String settlementNo;          // 结算单号
    private String trackingNo;             // 物流追踪号
    private String orderNo;                // 订单号
    private Long carrierId;                // 承运商ID
    private String carrierName;            // 承运商名称
    private BigDecimal freight;            // 运费金额
    private BigDecimal handlingFee;        // 手续费
    private BigDecimal discount;           // 折扣
    private BigDecimal totalAmount;        // 结算总金额
    private LogisticsSettlementStatus status;
    private String billingCycle;           // 结算周期: DAILY/WEEKLY/MONTHLY
    private LocalDateTime settlementDate;  // 结算日期
    private LocalDateTime paidAt;          // 支付日期
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum LogisticsSettlementStatus {
        PENDING,        // 待结算
        CONFIRMED,      // 已确认
        PROCESSING,     // 处理中
        COMPLETED,      // 已完成（已付款）
        FAILED,         // 失败
        CANCELLED       // 已取消
    }

    /**
     * 创建物流运费结算单（物流订单送达时自动调用）
     */
    public void createLogisticsSettlement(String settlementNo, String trackingNo, String orderNo,
                                           Long carrierId, String carrierName, BigDecimal freight,
                                           BigDecimal handlingFeeRate, BigDecimal discount) {
        this.settlementNo = settlementNo;
        this.trackingNo = trackingNo;
        this.orderNo = orderNo;
        this.carrierId = carrierId;
        this.carrierName = carrierName;
        this.freight = freight;
        this.handlingFee = freight.multiply(handlingFeeRate);
        this.discount = discount != null ? discount : BigDecimal.ZERO;
        this.totalAmount = freight.add(handlingFee).subtract(this.discount);
        this.status = LogisticsSettlementStatus.PENDING;
        this.billingCycle = "MONTHLY"; // 默认月结
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void confirm() {
        if (status != LogisticsSettlementStatus.PENDING) {
            throw new IllegalStateException("Can only confirm pending logistics settlements");
        }
        status = LogisticsSettlementStatus.CONFIRMED;
        updatedAt = LocalDateTime.now();
    }

    public void process() {
        if (status != LogisticsSettlementStatus.CONFIRMED) {
            throw new IllegalStateException("Can only process confirmed logistics settlements");
        }
        status = LogisticsSettlementStatus.PROCESSING;
        updatedAt = LocalDateTime.now();
    }

    public void complete() {
        if (status != LogisticsSettlementStatus.PROCESSING) {
            throw new IllegalStateException("Can only complete processing logistics settlements");
        }
        status = LogisticsSettlementStatus.COMPLETED;
        this.paidAt = LocalDateTime.now();
        this.settlementDate = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    public void fail(String reason) {
        if (status != LogisticsSettlementStatus.PROCESSING) {
            throw new IllegalStateException("Can only fail processing logistics settlements");
        }
        status = LogisticsSettlementStatus.FAILED;
        this.remark = reason;
        updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status != LogisticsSettlementStatus.PENDING && status != LogisticsSettlementStatus.CONFIRMED) {
            throw new IllegalStateException("Cannot cancel logistics settlements in current status");
        }
        status = LogisticsSettlementStatus.CANCELLED;
        updatedAt = LocalDateTime.now();
    }

    public void applyDiscount(BigDecimal additionalDiscount) {
        if (status == LogisticsSettlementStatus.COMPLETED || status == LogisticsSettlementStatus.CANCELLED) {
            throw new IllegalStateException("Cannot apply discount to completed or cancelled settlements");
        }
        this.discount = this.discount.add(additionalDiscount);
        this.totalAmount = freight.add(handlingFee).subtract(this.discount);
        updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getSettlementNo() { return settlementNo; }
    public String getTrackingNo() { return trackingNo; }
    public String getOrderNo() { return orderNo; }
    public Long getCarrierId() { return carrierId; }
    public String getCarrierName() { return carrierName; }
    public BigDecimal getFreight() { return freight; }
    public BigDecimal getHandlingFee() { return handlingFee; }
    public BigDecimal getDiscount() { return discount; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public LogisticsSettlementStatus getStatus() { return status; }
    public String getBillingCycle() { return billingCycle; }
    public LocalDateTime getSettlementDate() { return settlementDate; }
    public LocalDateTime getPaidAt() { return paidAt; }
    public String getRemark() { return remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setSettlementNo(String settlementNo) { this.settlementNo = settlementNo; }
    public void setTrackingNo(String trackingNo) { this.trackingNo = trackingNo; }
    public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
    public void setCarrierId(Long carrierId) { this.carrierId = carrierId; }
    public void setCarrierName(String carrierName) { this.carrierName = carrierName; }
    public void setFreight(BigDecimal freight) { this.freight = freight; }
    public void setHandlingFee(BigDecimal handlingFee) { this.handlingFee = handlingFee; }
    public void setDiscount(BigDecimal discount) { this.discount = discount; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public void setStatus(LogisticsSettlementStatus status) { this.status = status; }
    public void setBillingCycle(String billingCycle) { this.billingCycle = billingCycle; }
    public void setSettlementDate(LocalDateTime settlementDate) { this.settlementDate = settlementDate; }
    public void setPaidAt(LocalDateTime paidAt) { this.paidAt = paidAt; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}