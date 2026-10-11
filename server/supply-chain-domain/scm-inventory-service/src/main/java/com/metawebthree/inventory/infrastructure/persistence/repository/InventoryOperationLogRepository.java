package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.InventoryOperationLog;
import java.time.LocalDateTime;
import java.util.List;

public interface InventoryOperationLogRepository {
    /**
     * Save operation log
     */
    InventoryOperationLog save(InventoryOperationLog log);
    
    /**
     * Find logs by biz ID
     */
    List<InventoryOperationLog> findByBizId(String bizId);
    
    /**
     * Find logs by SKU code and warehouse
     */
    List<InventoryOperationLog> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId);
    
    /**
     * Find logs by operation type
     */
    List<InventoryOperationLog> findByOperationType(String operationType);
    
    /**
     * Find logs by date range
     */
    List<InventoryOperationLog> findByOperatedAtBetween(LocalDateTime start, LocalDateTime end);
    
    /**
     * Find logs by operator
     */
    List<InventoryOperationLog> findByOperatorId(String operatorId);
    
    /**
     * Find logs by result status
     */
    List<InventoryOperationLog> findByResult(String result);
}