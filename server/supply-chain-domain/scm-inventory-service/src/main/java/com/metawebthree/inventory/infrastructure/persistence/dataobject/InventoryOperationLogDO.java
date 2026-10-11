package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * Inventory Operation Log Data Object
 * Maps to inventory_operation_log table
 */
@Data
public class InventoryOperationLogDO {
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