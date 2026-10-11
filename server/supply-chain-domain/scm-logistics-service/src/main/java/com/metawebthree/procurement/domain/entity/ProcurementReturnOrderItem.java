package com.metawebthree.procurement.domain.entity;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class ProcurementReturnOrderItem {
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
    
    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_RETURNED = "RETURNED";
    public static final String STATUS_RECEIVED = "RECEIVED";
}