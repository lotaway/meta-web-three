package com.metawebthree.inventory.application.dto.stockcheck;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class StockCheckPlanDetailDTO {
    private Long id;
    private Long planId;
    private String skuCode;
    private String productName;
    private String locationCode;
    private BigDecimal bookQuantity;
    private BigDecimal checkQuantity;
    private String remark;
}