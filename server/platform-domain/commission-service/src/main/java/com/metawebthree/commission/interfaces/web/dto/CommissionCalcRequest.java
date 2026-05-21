package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Schema(description = "佣金计算请求")
public class CommissionCalcRequest {
    @Schema(description = "订单ID")
    private Long orderId;
    @Schema(description = "用户ID")
    private Long userId;
    @Schema(description = "支付金额")
    private BigDecimal payAmount;
    @Schema(description = "可提现时间")
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
