package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class OutboundOrder {
    private Long id;
    private String orderNo;
    private String outboundType;
    private Long warehouseId;
    private String relatedOrderNo;
    private String status;
    private String remark;
    private String operator;
    private String receiverName;
    private String receiverPhone;
    private String receiverAddress;
    private LocalDateTime planDepartureTime;
    private LocalDateTime actualDepartureTime;
    private LocalDateTime completedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<OutboundOrderItem> items;

    public void confirm() {
        if (!"PENDING".equals(status)) {
            throw new IllegalStateException("Only pending order can be confirmed");
        }
        this.status = "CONFIRMED";
    }

    public void dispatch() {
        if (!"CONFIRMED".equals(status)) {
            throw new IllegalStateException("Only confirmed order can be dispatched");
        }
        this.status = "DISPATCHED";
        this.actualDepartureTime = LocalDateTime.now();
    }

    public void complete() {
        if (!"DISPATCHED".equals(status)) {
            throw new IllegalStateException("Only dispatched order can be completed");
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