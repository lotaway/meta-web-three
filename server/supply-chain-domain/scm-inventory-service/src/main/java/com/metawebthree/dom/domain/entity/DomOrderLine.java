package com.metawebthree.dom.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class DomOrderLine {
    private Long id;
    private Long domOrderId;
    private String skuCode;
    private String skuName;
    private Integer quantity;
    private Integer fulfilledQuantity;
    private Long warehouseId;
    private String warehouseName;
    private BigDecimal unitPrice;
    private DomOrderLineStatus status;
    private LocalDateTime createdAt;
}
