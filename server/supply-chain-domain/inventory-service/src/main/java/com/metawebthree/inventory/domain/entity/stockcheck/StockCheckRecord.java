package com.metawebthree.inventory.domain.entity.stockcheck;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 盘点记录实体
 * 记录每次实际盘点的数据
 */
@Data
public class StockCheckRecord {
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
    private String sourceSystem;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;
    private Integer version;

    public static final String DIFF_TYPE_NONE = "NONE";
    public static final String DIFF_TYPE_SHORT = "SHORT";
    public static final String DIFF_TYPE_OVER = "OVER";

    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_CONFIRMED = "CONFIRMED";
    public static final String STATUS_ADJUSTED = "ADJUSTED";

    public void calculateDifference() {
        if (bookQuantity != null && checkQuantity != null) {
            differenceQuantity = checkQuantity.subtract(bookQuantity);
            if (differenceQuantity.compareTo(BigDecimal.ZERO) > 0) {
                differenceType = DIFF_TYPE_OVER;
            } else if (differenceQuantity.compareTo(BigDecimal.ZERO) < 0) {
                differenceType = DIFF_TYPE_SHORT;
            } else {
                differenceType = DIFF_TYPE_NONE;
            }
        }
    }

    public boolean hasDifference() {
        return differenceQuantity != null && differenceQuantity.compareTo(BigDecimal.ZERO) != 0;
    }
}