package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.InventoryRecord;
import java.util.List;

public interface InventoryRecordRepository {
    List<InventoryRecord> findByWarehouseAndDateRange(Long warehouseId, java.time.LocalDateTime startDate, java.time.LocalDateTime endDate);
    List<InventoryRecord> findBySkuCodeAndDateRange(String skuCode, Long warehouseId, java.time.LocalDateTime startDate, java.time.LocalDateTime endDate);
}