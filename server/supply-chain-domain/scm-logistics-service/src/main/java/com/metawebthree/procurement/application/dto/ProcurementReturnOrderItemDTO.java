package com.metawebthree.procurement.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class ProcurementReturnOrderItemDTO {
    private Long id;
    private Long returnOrderId;
    private String returnNo;
    private String sourceOrderNo;
    private String sourceOrderItemId;
    private String skuCode;
    private String productName;
    private Integer returnQuantity;
    private BigDecimal unitPrice;
    private BigDecimal totalAmount;
    private String reason;
    private String status;
}