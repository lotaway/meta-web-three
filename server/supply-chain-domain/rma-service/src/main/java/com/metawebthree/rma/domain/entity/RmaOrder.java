package com.metawebthree.rma.domain.entity;

import com.metawebthree.rma.domain.RmaOrderStatus;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class RmaOrder {
    private Long id;
    private String rmaNo;
    private String orderNo;
    private String returnType;
    private RmaOrderStatus status;
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
        return RmaOrderStatus.PENDING == status;
    }

    public boolean canDispose() {
        return RmaOrderStatus.INSPECTED == status;
    }

    public boolean canComplete() {
        return RmaOrderStatus.DISPOSED == status;
    }

    public boolean canCancel() {
        return RmaOrderStatus.PENDING == status || RmaOrderStatus.AWAITING_INSPECTION == status;
    }
}
