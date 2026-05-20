package com.metawebthree.procurement.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class ProcurementOrder {
    private Long id;
    private String orderNo;
    private String supplierCode;
    private Long warehouseId;
    private String purchaseType;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private String paymentTerms;
    private String deliveryTerms;
    private String remark;
    private String approver;
    private LocalDateTime approvedAt;
    private LocalDateTime expectedDeliveryDate;
    private LocalDateTime actualDeliveryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void approve(String approver) {
        this.status = "APPROVED";
        this.approver = approver;
        this.approvedAt = LocalDateTime.now();
    }

    public void reject(String reason) {
        this.status = "REJECTED";
    }

    public void complete() {
        this.status = "COMPLETED";
        this.actualDeliveryDate = LocalDateTime.now();
    }

    public void cancel() {
        if ("PENDING".equals(status) || "DRAFT".equals(status)) {
            this.status = "CANCELLED";
        }
    }
}