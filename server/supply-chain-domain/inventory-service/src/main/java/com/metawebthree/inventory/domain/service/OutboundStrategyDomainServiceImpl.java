package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.InventoryBatch;
import com.metawebthree.inventory.domain.repository.InventoryBatchRepository;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class OutboundStrategyDomainServiceImpl implements OutboundStrategyDomainService {

    private final InventoryBatchRepository inventoryBatchRepository;

    public OutboundStrategyDomainServiceImpl(InventoryBatchRepository inventoryBatchRepository) {
        this.inventoryBatchRepository = inventoryBatchRepository;
    }

    @Override
    public List<InventoryBatch> selectBatchesForOutbound(String skuCode, Long warehouseId, Integer quantity, String strategyType, String specificBatchNo) {
        // 查询可用的库存批次
        List<InventoryBatch> availableBatches = inventoryBatchRepository.findAvailableBatches(skuCode, warehouseId);
        
        // 根据策略选择批次
        List<InventoryBatch> sortedBatches = selectBatchesByStrategy(availableBatches, strategyType, specificBatchNo);
        
        // 分配数量
        return allocateQuantity(sortedBatches, quantity);
    }

    public List<InventoryBatch> selectBatchesByStrategy(List<InventoryBatch> availableBatches, String strategyType, String specificBatchNo) {
        if (availableBatches == null || availableBatches.isEmpty()) {
            return List.of();
        }
        List<InventoryBatch> validBatches = availableBatches.stream()
                .filter(InventoryBatch::isAvailable)
                .filter(batch -> !batch.isExpired())
                .collect(Collectors.toList());

        if (validBatches.isEmpty()) {
            return List.of();
        }

        if ("SPECIFIC_BATCH".equals(strategyType) && specificBatchNo != null) {
            return validBatches.stream()
                    .filter(batch -> specificBatchNo.equals(batch.getBatchNo()))
                    .collect(Collectors.toList());
        }

        if ("LIFO".equals(strategyType)) {
            return validBatches.stream()
                    .sorted(Comparator.comparing(InventoryBatch::getInboundDate).reversed()
                            .thenComparing(Comparator.comparing(InventoryBatch::getId).reversed()))
                    .collect(Collectors.toList());
        }

        return validBatches.stream()
                .sorted(Comparator.comparing(InventoryBatch::getInboundDate)
                        .thenComparing(Comparator.comparing(InventoryBatch::getId)))
                .collect(Collectors.toList());
    }

    public List<InventoryBatch> allocateQuantity(List<InventoryBatch> sortedBatches, Integer requiredQuantity) {
        if (sortedBatches == null || sortedBatches.isEmpty() || requiredQuantity == null || requiredQuantity <= 0) {
            return List.of();
        }

        List<InventoryBatch> allocated = new java.util.ArrayList<>();
        int remaining = requiredQuantity;

        for (InventoryBatch batch : sortedBatches) {
            if (remaining <= 0) {
                break;
            }
            int available = batch.getAvailableQuantity();
            int toAllocate = Math.min(available, remaining);

            InventoryBatch allocatedBatch = new InventoryBatch();
            allocatedBatch.setId(batch.getId());
            allocatedBatch.setSkuCode(batch.getSkuCode());
            allocatedBatch.setWarehouseId(batch.getWarehouseId());
            allocatedBatch.setBatchNo(batch.getBatchNo());
            allocatedBatch.setQuantity(toAllocate);
            allocatedBatch.setAvailableQuantity(toAllocate);
            allocatedBatch.setInboundDate(batch.getInboundDate());
            allocatedBatch.setExpiryDate(batch.getExpiryDate());
            allocatedBatch.setUnitCost(batch.getUnitCost());
            allocatedBatch.setLocationCode(batch.getLocationCode());

            allocated.add(allocatedBatch);
            remaining -= toAllocate;
        }

        if (remaining > 0) {
            throw new IllegalStateException("Insufficient inventory quantity in all batches");
        }

        return allocated;
    }
}