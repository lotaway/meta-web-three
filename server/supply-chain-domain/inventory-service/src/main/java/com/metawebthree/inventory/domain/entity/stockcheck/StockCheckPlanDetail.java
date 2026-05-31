package com.metawebthree.inventory.domain.entity.stockcheck;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 盘点计划明细
 */
@Data
public class StockCheckPlanDetail {
    private Long id;
    private Long planId;
    private String skuCode;
    private String productName;
    private String locationCode;
    private BigDecimal bookQuantity;
    private BigDecimal checkQuantity;
    private String remark;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
    private Boolean deleted;
    private Integer version;
}