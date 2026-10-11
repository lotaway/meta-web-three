package com.metawebthree.rma.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class RmaDisposition {
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String dispositionType;
    private BigDecimal refundAmount;
    private String replacementSkuCode;
    private Integer replacementQuantity;
    private Integer scrapQuantity;
    private String scrapReason;
    private String dispositionBy;
    private LocalDateTime dispositionDate;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
