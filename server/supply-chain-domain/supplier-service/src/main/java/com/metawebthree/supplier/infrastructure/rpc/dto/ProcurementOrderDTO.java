package com.metawebthree.supplier.infrastructure.rpc.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 采购订单DTO（用于RPC调用）
 * 复制自 procurement-service，用于跨服务调用
 */
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
    private LocalDateTime actualDeliveryDate;
    private LocalDateTime createdAt;
}