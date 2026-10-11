package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class Inventory {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer totalQuantity;
    private Integer availableQuantity;
    private Integer reservedQuantity;
    private Integer defectiveQuantity;
    private BigDecimal unitCost;
    private Integer safetyStock;
    private Integer leadTimeDays;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public boolean needsReplenishment() {
        return availableQuantity < safetyStock;
    }

    public Integer calculateReorderQuantity(Integer averageDailySales) {
        if (leadTimeDays == null || leadTimeDays <= 0 || averageDailySales == null || averageDailySales <= 0) {
            return 0;
        }
        Integer leadTimeDemand = averageDailySales * leadTimeDays;
        Integer targetStock = safetyStock + leadTimeDemand;
        Integer reorderQty = targetStock - availableQuantity;
        return Math.max(0, reorderQty);
    }

    public boolean canReserve(Integer quantity) {
        return availableQuantity >= quantity;
    }

    public void reserve(Integer quantity) {
        if (!canReserve(quantity)) {
            throw new IllegalStateException("Insufficient inventory");
        }
        availableQuantity -= quantity;
        reservedQuantity += quantity;
    }

    public void confirmReserve(Integer quantity) {
        reservedQuantity -= quantity;
        totalQuantity -= quantity;
    }

    public void cancelReserve(Integer quantity) {
        reservedQuantity -= quantity;
        availableQuantity += quantity;
    }

    public void increase(Integer quantity) {
        totalQuantity += quantity;
        availableQuantity += quantity;
    }

    public void decrease(Integer quantity) {
        if (totalQuantity < quantity) {
            throw new IllegalStateException("Insufficient total inventory");
        }
        totalQuantity -= quantity;
        if (availableQuantity >= quantity) {
            availableQuantity -= quantity;
        } else {
            Integer used = quantity - availableQuantity;
            availableQuantity = 0;
            reservedQuantity -= used;
        }
    }
}