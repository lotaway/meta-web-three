package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryRecord {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private String bizType;
    private String bizId;
    private Integer quantity;
    private Integer beforeQuantity;
    private Integer afterQuantity;
    private String remark;
    private String operator;
    private LocalDateTime createdAt;
}