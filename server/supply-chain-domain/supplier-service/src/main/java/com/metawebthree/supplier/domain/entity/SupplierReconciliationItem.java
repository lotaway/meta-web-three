package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class SupplierReconciliationItem {
    private Long id;
    private Long reconciliationId;
    private String orderNo;
    private LocalDate orderDate;
    private LocalDate shippedDate;
    private BigDecimal invoicedAmount;
    private BigDecimal settledAmount;
    private BigDecimal pendingAmount;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void calculatePending() {
        if (invoicedAmount != null && settledAmount != null) {
            this.pendingAmount = invoicedAmount.subtract(settledAmount);
        }
    }
}