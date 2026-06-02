package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class StocktakeOrderItem {
    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_COUNTED = "COUNTED";
    public static final String STATUS_CONFIRMED = "CONFIRMED";
    public static final String STATUS_ADJUSTED = "ADJUSTED";

    private Long id;
    private Long stocktakeOrderId;
    private String skuCode;
    private String skuName;
    private String unit;
    private BigDecimal systemQuantity;
    private BigDecimal countedQuantity;
    private BigDecimal discrepancyQuantity;
    private BigDecimal discrepancyAmount;
    private String discrepancyReason;
    private String status;
    private String counter;
    private LocalDateTime countedAt;
    private String checker;
    private LocalDateTime checkedAt;
    private String adjuster;
    private LocalDateTime adjustedAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}