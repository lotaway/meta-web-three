package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReplenishmentRecommendationDTO {
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
    private Boolean isUrgent;
    private LocalDateTime generatedAt;
}