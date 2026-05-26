package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class InboundOrderItem {
    private Long id;
    private Long orderId;
    private String skuCode;
    private String productName;
    private Integer planQuantity;
    private Integer actualQuantity;
    private Long locationId;
    private String status;
    private BigDecimal unitCost;
    private String batchNo;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}