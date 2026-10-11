package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReplenishmentRecommendation {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer currentStock;
    private Integer safetyStock;
    private Integer leadTimeDays;
    private Integer averageDailySales;
    private Integer recommendedQuantity;
    private String recommendationType;
    private String status;
    private LocalDateTime generatedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void calculateRecommendation() {
        if (leadTimeDays == null || leadTimeDays <= 0 || averageDailySales == null || averageDailySales <= 0) {
            recommendedQuantity = 0;
            recommendationType = "MANUAL";
            return;
        }
        Integer leadTimeDemand = averageDailySales * leadTimeDays;
        Integer targetStock = safetyStock + leadTimeDemand;
        Integer calculatedQty = targetStock - currentStock;
        recommendedQuantity = Math.max(0, calculatedQty);
        recommendationType = "AUTO";
    }

    public boolean isUrgent() {
        if (currentStock == null || safetyStock == null) {
            return false;
        }
        return currentStock < safetyStock;
    }

    public void generate() {
        calculateRecommendation();
        status = "PENDING";
        generatedAt = LocalDateTime.now();
    }

    public void approve() {
        status = "APPROVED";
    }

    public void reject() {
        status = "REJECTED";
    }
}