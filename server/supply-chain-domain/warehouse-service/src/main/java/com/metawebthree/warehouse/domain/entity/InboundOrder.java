package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class InboundOrder {
    private Long id;
    private String orderNo;
    private String inboundType;
    private Long warehouseId;
    private String supplierCode;
    private String status;
    private String remark;
    private String operator;
    private LocalDateTime planArrivalTime;
    private LocalDateTime actualArrivalTime;
    private LocalDateTime completedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<InboundOrderItem> items;

    public void confirm() {
        if (!"PENDING".equals(status)) {
            throw new IllegalStateException("Only pending order can be confirmed");
        }
        this.status = "CONFIRMED";
    }

    public void complete() {
        if (!"CONFIRMED".equals(status)) {
            throw new IllegalStateException("Only confirmed order can be completed");
        }
        this.status = "COMPLETED";
        this.completedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (!"PENDING".equals(status) && !"CONFIRMED".equals(status)) {
            throw new IllegalStateException("Order cannot be cancelled");
        }
        this.status = "CANCELLED";
    }
}