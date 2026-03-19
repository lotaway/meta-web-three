package com.metawebthree.promotion.interfaces.web.dto;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CouponView {
    private String code;
    private Long couponTypeId;
    private String couponTypeName;
    private BigDecimal minimumOrderAmount;
    private BigDecimal discountAmount;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer useStatus;
    private Integer transferStatus;
    private Integer acquireMethod;

    public String getCode() { return code; }
    public void setCode(String code) { this.code = code; }
    public Long getCouponTypeId() { return couponTypeId; }
    public void setCouponTypeId(Long couponTypeId) { this.couponTypeId = couponTypeId; }
    public String getCouponTypeName() { return couponTypeName; }
    public void setCouponTypeName(String couponTypeName) { this.couponTypeName = couponTypeName; }
    public BigDecimal getMinimumOrderAmount() { return minimumOrderAmount; }
    public void setMinimumOrderAmount(BigDecimal minimumOrderAmount) { this.minimumOrderAmount = minimumOrderAmount; }
    public BigDecimal getDiscountAmount() { return discountAmount; }
    public void setDiscountAmount(BigDecimal discountAmount) { this.discountAmount = discountAmount; }
    public LocalDateTime getStartTime() { return startTime; }
    public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    public Integer getUseStatus() { return useStatus; }
    public void setUseStatus(Integer useStatus) { this.useStatus = useStatus; }
    public Integer getTransferStatus() { return transferStatus; }
    public void setTransferStatus(Integer transferStatus) { this.transferStatus = transferStatus; }
    public Integer getAcquireMethod() { return acquireMethod; }
    public void setAcquireMethod(Integer acquireMethod) { this.acquireMethod = acquireMethod; }
}
