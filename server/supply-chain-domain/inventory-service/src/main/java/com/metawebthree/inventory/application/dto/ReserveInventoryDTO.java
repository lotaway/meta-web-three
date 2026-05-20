package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class ReserveInventoryDTO {
    private String skuCode;
    private Long warehouseId;
    private Integer quantity;
    private String bizId;
    private String bizType;
    private String remark;
}