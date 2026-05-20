package com.metawebthree.procurement.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class ProcurementOrderDTO {
    private Long id;
    private String orderNo;
    private String supplierCode;
    private String supplierName;
    private Long warehouseId;
    private String warehouseName;
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
    private LocalDateTime createdAt;
}