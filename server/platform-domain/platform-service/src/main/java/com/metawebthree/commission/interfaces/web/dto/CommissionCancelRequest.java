package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "佣金取消请求")
public class CommissionCancelRequest {
    @Schema(description = "订单ID")
    private Long orderId;

    public Long getOrderId() { return orderId; }
    public void setOrderId(Long orderId) { this.orderId = orderId; }
}
