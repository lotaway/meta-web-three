package com.metawebthree.inventory.application.dto.stockcheck;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class StockCheckRecordDTO {
    private Long id;
    private Long planId;
    private String planNo;
    private String skuCode;
    private String productName;
    private String locationCode;
    private Long warehouseId;
    private BigDecimal bookQuantity;
    private BigDecimal checkQuantity;
    private BigDecimal differenceQuantity;
    private String differenceType;
    private String status;
    private String checker;
    private LocalDateTime checkTime;
    private String remark;
}