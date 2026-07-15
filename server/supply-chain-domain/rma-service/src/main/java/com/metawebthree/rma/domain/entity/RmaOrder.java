package com.metawebthree.rma.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class RmaOrder {
    private Long id;
    private String rmaNo;
    private String orderNo;
    private String returnType;
    private String status;
    private Long customerId;
    private String customerName;
    private String contactPhone;
    private String reasonCode;
    private String reasonDescription;
    private Long warehouseId;
    private Integer totalQuantity;
    private BigDecimal totalAmount;
    private String currency;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public boolean canInspect() {
        return "PENDING".equals(status);
    }

    public boolean canDispose() {
        return "INSPECTED".equals(status);
    }

    public boolean canComplete() {
        return "DISPOSED".equals(status);
    }

    public boolean canCancel() {
        return "PENDING".equals(status) || "AWAITING_INSPECTION".equals(status);
    }
}
