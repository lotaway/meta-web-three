package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Schema(description = "佣金记录视图")
public class CommissionRecordView {
    @Schema(description = "记录ID")
    private Long id;
    @Schema(description = "订单ID")
    private Long orderId;
    @Schema(description = "来源用户ID")
    private Long fromUserId;
    @Schema(description = "层级")
    private Integer level;
    @Schema(description = "金额")
    private BigDecimal amount;
    @Schema(description = "状态")
    private String status;
    @Schema(description = "可提现时间")
    private LocalDateTime availableAt;
    @Schema(description = "创建时间")
    private LocalDateTime createdAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getOrderId() { return orderId; }
    public void setOrderId(Long orderId) { this.orderId = orderId; }
    public Long getFromUserId() { return fromUserId; }
    public void setFromUserId(Long fromUserId) { this.fromUserId = fromUserId; }
    public Integer getLevel() { return level; }
    public void setLevel(Integer level) { this.level = level; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getAvailableAt() { return availableAt; }
    public void setAvailableAt(LocalDateTime availableAt) { this.availableAt = availableAt; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
