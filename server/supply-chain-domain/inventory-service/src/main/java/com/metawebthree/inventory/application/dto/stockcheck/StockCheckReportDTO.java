package com.metawebthree.inventory.application.dto.stockcheck;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class StockCheckReportDTO {
    private Long warehouseId;
    private String warehouseName;
    private String planNo;
    private String planName;
    private String checkType;
    
    private Integer totalSkus;
    private Integer checkedSkus;
    private Integer differenceCount;
    private Integer shortCount;
    private Integer overCount;
    
    private BigDecimal totalBookQuantity;
    private BigDecimal totalCheckQuantity;
    private BigDecimal totalDifferenceQuantity;
    
    private Integer pendingApprovalCount;
    private Integer approvedCount;
    private Integer rejectedCount;
    private Integer processedCount;
    
    private String completionRate;
    private String accuracyRate;
}