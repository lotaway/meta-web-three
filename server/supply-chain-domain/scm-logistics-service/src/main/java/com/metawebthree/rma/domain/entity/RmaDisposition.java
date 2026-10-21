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
    /**
     * 幂等键：同一 {@code rmaId + dispositionType + refundAmount} 的重复请求复用同一键，
     * 用于去重，避免重复回库/重复退款产生副作用。
     */
    private String idempotentKey;
    /**
     * 结算状态：PENDING / SUCCESS / FAILED / ERROR。
     */
    private String status;
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
