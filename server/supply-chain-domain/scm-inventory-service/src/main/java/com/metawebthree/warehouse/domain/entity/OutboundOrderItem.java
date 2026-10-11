package com.metawebthree.warehouse.domain.entity;

import lombok.Data;

@Data
public class OutboundOrderItem {
    private Long id;
    private Long orderId;
    private String skuCode;
    private String productName;
    private Integer quantity;
    private Long locationId;
    private String status;
}