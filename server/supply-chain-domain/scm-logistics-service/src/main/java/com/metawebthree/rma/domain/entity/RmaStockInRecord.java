package com.metawebthree.rma.domain.entity;

import lombok.Data;

import java.time.LocalDateTime;

/**
 * Stock-in ledger entry for an accepted RMA line.
 *
 * <p>One row per {@code rmaId + skuCode}. Its {@code status} makes a failed
 * stock-in retryable and traceable, and acts as the local idempotency guard so
 * a successful line is never sent to the inventory service again.</p>
 */
@Data
public class RmaStockInRecord {
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String skuCode;
    private String skuName;
    private Long warehouseId;
    private Integer quantity;
    /** PENDING / SUCCESS / FAILED */
    private String status;
    private Integer retryCount;
    private String lastError;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
