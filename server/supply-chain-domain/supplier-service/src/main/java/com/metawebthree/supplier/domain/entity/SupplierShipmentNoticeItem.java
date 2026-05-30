package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class SupplierShipmentNoticeItem {
    private Long id;
    private Long noticeId;
    private String productCode;
    private String productName;
    private String unit;
    private BigDecimal quantity;
    private BigDecimal weight;
    private BigDecimal volume;
    private String batchNo;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}