package com.metawebthree.logistics.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class LogisticsOrder {
    private Long id;
    private String trackingNo;
    private String orderNo;
    private Long carrierId;
    private String carrierName;
    private String serviceType;
    private String senderName;
    private String senderPhone;
    private String senderProvince;
    private String senderCity;
    private String senderDistrict;
    private String senderAddress;
    private String receiverName;
    private String receiverPhone;
    private String receiverProvince;
    private String receiverCity;
    private String receiverDistrict;
    private String receiverAddress;
    private BigDecimal weight;
    private BigDecimal volume;
    private BigDecimal freight;
    private String status;
    private LocalDateTime pickedUpAt;
    private LocalDateTime inTransitAt;
    private LocalDateTime outForDeliveryAt;
    private LocalDateTime deliveredAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void pickUp() {
        this.status = "PICKED_UP";
        this.pickedUpAt = LocalDateTime.now();
    }

    public void inTransit() {
        this.status = "IN_TRANSIT";
        this.inTransitAt = LocalDateTime.now();
    }

    public void outForDelivery() {
        this.status = "OUT_FOR_DELIVERY";
        this.outForDeliveryAt = LocalDateTime.now();
    }

    public void delivered() {
        this.status = "DELIVERED";
        this.deliveredAt = LocalDateTime.now();
    }

    public void exception(String reason) {
        this.status = "EXCEPTION";
    }
}