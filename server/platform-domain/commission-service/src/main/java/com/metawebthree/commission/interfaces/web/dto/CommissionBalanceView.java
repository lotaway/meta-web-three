package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;

@Schema(description = "佣金余额视图")
public class CommissionBalanceView {
    @Schema(description = "用户ID")
    private Long userId;
    @Schema(description = "总金额")
    private BigDecimal totalAmount;
    @Schema(description = "可用金额")
    private BigDecimal availableAmount;
    @Schema(description = "冻结金额")
    private BigDecimal frozenAmount;

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public BigDecimal getAvailableAmount() { return availableAmount; }
    public void setAvailableAmount(BigDecimal availableAmount) { this.availableAmount = availableAmount; }
    public BigDecimal getFrozenAmount() { return frozenAmount; }
    public void setFrozenAmount(BigDecimal frozenAmount) { this.frozenAmount = frozenAmount; }
}
