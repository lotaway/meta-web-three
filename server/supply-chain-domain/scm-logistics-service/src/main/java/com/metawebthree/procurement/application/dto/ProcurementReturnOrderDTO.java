package com.metawebthree.procurement.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class ProcurementReturnOrderDTO {
    private Long id;
    private String returnNo;
    private String sourceOrderNo;
    private String sourceOrderType;
    private String supplierCode;
    private String supplierName;
    private Long warehouseId;
    private String warehouseName;
    private String returnType;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private String reason;
    private String remark;
    private String approver;
    private String approvalComment;
    private LocalDateTime approvedAt;
    private LocalDateTime expectedReturnDate;
    private LocalDateTime actualReturnDate;
    private String logisticsCompany;
    private String trackingNumber;
    private LocalDateTime shippedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<ProcurementReturnOrderItemDTO> items;
}