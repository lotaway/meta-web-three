package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.InventoryBatch;
import java.util.List;

public interface OutboundStrategyDomainService {
    List<InventoryBatch> selectBatchesForOutbound(String skuCode, Long warehouseId, Integer quantity, String strategyType, String specificBatchNo);
    
    List<InventoryBatch> selectBatchesByStrategy(List<InventoryBatch> availableBatches, String strategyType, String specificBatchNo);
    
    List<InventoryBatch> allocateQuantity(List<InventoryBatch> sortedBatches, Integer requiredQuantity);
}