package com.metawebthree.commission.interfaces.web.dto;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CommissionCalcRequest {
    private Long orderId;
    private Long userId;
    private BigDecimal payAmount;
    private LocalDateTime availableAt;

    public Long getOrderId() { return orderId; }
    public void setOrderId(Long orderId) { this.orderId = orderId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public BigDecimal getPayAmount() { return payAmount; }
    public void setPayAmount(BigDecimal payAmount) { this.payAmount = payAmount; }
    public LocalDateTime getAvailableAt() { return availableAt; }
    public void setAvailableAt(LocalDateTime availableAt) { this.availableAt = availableAt; }
}
