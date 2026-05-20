package com.metawebthree.promotion.interfaces.web.dto;

import java.math.BigDecimal;

public class CouponValidateResponse {
    private String couponTypeName;
    private BigDecimal discountAmount;
    private BigDecimal payableAmount;

    public String getCouponTypeName() { return couponTypeName; }
    public void setCouponTypeName(String couponTypeName) { this.couponTypeName = couponTypeName; }
    public BigDecimal getDiscountAmount() { return discountAmount; }
    public void setDiscountAmount(BigDecimal discountAmount) { this.discountAmount = discountAmount; }
    public BigDecimal getPayableAmount() { return payableAmount; }
    public void setPayableAmount(BigDecimal payableAmount) { this.payableAmount = payableAmount; }
}
