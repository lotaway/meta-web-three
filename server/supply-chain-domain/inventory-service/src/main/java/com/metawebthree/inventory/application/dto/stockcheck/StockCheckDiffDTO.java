package com.metawebthree.inventory.application.dto.stockcheck;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class StockCheckDiffDTO {
    private Long id;
    private Long recordId;
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
    private String processingStatus;
    private String approvalStatus;
    private String approver;
    private LocalDateTime approvalTime;
    private String approvalRemark;
    private String solution;
    private String processor;
    private LocalDateTime processTime;
    private String processRemark;
    private Boolean needsApproval;
}