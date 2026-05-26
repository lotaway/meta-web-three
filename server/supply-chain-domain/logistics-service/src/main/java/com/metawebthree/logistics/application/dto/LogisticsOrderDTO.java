package com.metawebthree.logistics.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class LogisticsOrderDTO {
    private Long id;
    private String trackingNo;
    private String orderNo;
    private Long carrierId;
    private String carrierName;
    private String serviceType;
    private String senderName;
    private String senderPhone;
    private String senderAddress;
    private String senderProvince;
    private String senderCity;
    private String senderDistrict;
    private String receiverName;
    private String receiverPhone;
    private String receiverAddress;
    private String receiverProvince;
    private String receiverCity;
    private String receiverDistrict;
    private BigDecimal weight;
    private BigDecimal volume;
    private BigDecimal freight;
    private String status;
    private LocalDateTime pickedUpAt;
    private LocalDateTime deliveredAt;
    private LocalDateTime createdAt;
}