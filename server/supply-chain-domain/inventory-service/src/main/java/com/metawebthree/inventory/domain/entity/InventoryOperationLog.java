package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryOperationLog {
    private Long id;
    private String operationType;
    private String skuCode;
    private Long warehouseId;
    private Integer quantity;
    private String bizId;
    private String remark;
    private String operatorId;
    private String operatorName;
    private Integer quantityBefore;
    private Integer quantityAfter;
    private LocalDateTime operatedAt;
    private String result;
    private String errorMessage;
    private String requestId;
    private String clientIp;
}