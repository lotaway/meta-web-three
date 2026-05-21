package com.metawebthree.commission.domain;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CommissionAccount {
    private Long id;
    private Long userId;
    private BigDecimal totalAmount;
    private BigDecimal availableAmount;
    private BigDecimal frozenAmount;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public BigDecimal getAvailableAmount() { return availableAmount; }
    public void setAvailableAmount(BigDecimal availableAmount) { this.availableAmount = availableAmount; }
    public BigDecimal getFrozenAmount() { return frozenAmount; }
    public void setFrozenAmount(BigDecimal frozenAmount) { this.frozenAmount = frozenAmount; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
