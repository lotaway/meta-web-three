package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class InventoryBatch {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private String batchNo;
    private Integer quantity;
    private Integer availableQuantity;
    private Integer reservedQuantity;
    private Integer pickedQuantity;
    private LocalDateTime inboundDate;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private BigDecimal unitCost;
    private String locationCode;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public boolean isExpired() {
        if (expiryDate == null) {
            return false;
        }
        return LocalDateTime.now().isAfter(expiryDate);
    }

    public boolean isAvailable() {
        return "AVAILABLE".equals(status) && availableQuantity > 0;
    }

    public boolean canPick(Integer quantity) {
        return "AVAILABLE".equals(status) && availableQuantity >= quantity;
    }

    public void pick(Integer quantity) {
        if (!canPick(quantity)) {
            throw new IllegalStateException("Batch insufficient available quantity");
        }
        availableQuantity -= quantity;
        pickedQuantity += quantity;
    }

    public void confirmPick(Integer quantity) {
        pickedQuantity -= quantity;
        quantity -= quantity;
    }

    public void cancelPick(Integer quantity) {
        pickedQuantity -= quantity;
        availableQuantity += quantity;
    }

    public void reserve(Integer quantity) {
        if (availableQuantity < quantity) {
            throw new IllegalStateException("Insufficient available quantity in batch");
        }
        availableQuantity -= quantity;
        reservedQuantity += quantity;
    }

    public void cancelReserve(Integer quantity) {
        reservedQuantity -= quantity;
        availableQuantity += quantity;
    }

    public void confirmReserve(Integer quantity) {
        reservedQuantity -= quantity;
    }
}