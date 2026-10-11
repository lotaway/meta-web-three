package com.metawebthree.inventory.domain.repository;

import com.metawebthree.inventory.domain.entity.InventoryBatch;
import java.util.List;

public interface InventoryBatchRepository {
    InventoryBatch findById(Long id);
    InventoryBatch findByBatchNo(String skuCode, Long warehouseId, String batchNo);
    List<InventoryBatch> findAvailableBatches(String skuCode, Long warehouseId);
    List<InventoryBatch> findByWarehouseAndSku(Long warehouseId, String skuCode);
    InventoryBatch save(InventoryBatch batch);
    boolean update(InventoryBatch batch);
    boolean delete(Long id);
}