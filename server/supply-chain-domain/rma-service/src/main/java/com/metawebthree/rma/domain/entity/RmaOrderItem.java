package com.metawebthree.rma.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class RmaOrderItem {
    private Long id;
    private Long rmaId;
    private String skuCode;
    private String skuName;
    private Integer expectedQuantity;
    private Integer inspectedQuantity;
    private Integer acceptedQuantity;
    private BigDecimal unitPrice;
    private String reasonCode;
    private String reasonDescription;
    private LocalDateTime createdAt;
}
