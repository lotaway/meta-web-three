package com.metawebthree.supplier.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class SupplierReconciliationDTO {
    private Long id;
    private String reconciliationNo;
    private String supplierCode;
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private Integer orderCount;
    private BigDecimal totalAmount;
    private BigDecimal shippedAmount;
    private BigDecimal invoicedAmount;
    private BigDecimal settledAmount;
    private BigDecimal pendingAmount;
    private String currency;
    private String status;
    private LocalDateTime submittedAt;
    private LocalDateTime confirmedAt;
    private String confirmedBy;
    private LocalDateTime paidAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<SupplierReconciliationItemDTO> items;

    @Data
    public static class SupplierReconciliationItemDTO {
        private Long id;
        private String orderNo;
        private LocalDate orderDate;
        private LocalDate shippedDate;
        private BigDecimal invoicedAmount;
        private BigDecimal settledAmount;
        private BigDecimal pendingAmount;
        private String status;
        private String remark;
    }
}