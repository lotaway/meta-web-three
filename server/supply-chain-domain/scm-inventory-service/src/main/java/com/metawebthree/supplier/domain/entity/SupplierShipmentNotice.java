package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Data
public class SupplierShipmentNotice {
    private Long id;
    private String noticeNo;
    private String supplierCode;
    private String orderNo;
    private Long warehouseId;
    private LocalDateTime expectedShipmentDate;
    private LocalDateTime actualShipmentDate;
    private String shipmentMethod;
    private String carrierName;
    private String carrierContact;
    private String trackingNumber;
    private String vehicleNumber;
    private String driverName;
    private String driverPhone;
    private BigDecimal totalQuantity;
    private BigDecimal totalWeight;
    private BigDecimal totalVolume;
    private String status;
    private String remark;
    private String confirmer;
    private LocalDateTime confirmedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<SupplierShipmentNoticeItem> items = new ArrayList<>();

    public void submit() {
        if ("DRAFT".equals(status)) {
            this.status = "SUBMITTED";
        }
    }

    public void confirm(String confirmer) {
        if ("SUBMITTED".equals(status)) {
            this.status = "CONFIRMED";
            this.confirmer = confirmer;
            this.confirmedAt = LocalDateTime.now();
        }
    }

    public void inTransit() {
        if ("CONFIRMED".equals(status)) {
            this.status = "IN_TRANSIT";
            this.actualShipmentDate = LocalDateTime.now();
        }
    }

    public void delivered() {
        if ("IN_TRANSIT".equals(status)) {
            this.status = "DELIVERED";
        }
    }

    public void cancel() {
        if ("DRAFT".equals(status) || "SUBMITTED".equals(status)) {
            this.status = "CANCELLED";
        }
    }

    public boolean canEdit() {
        return "DRAFT".equals(status);
    }

    public boolean canSubmit() {
        return "DRAFT".equals(status);
    }

    public boolean canConfirm() {
        return "SUBMITTED".equals(status);
    }
}