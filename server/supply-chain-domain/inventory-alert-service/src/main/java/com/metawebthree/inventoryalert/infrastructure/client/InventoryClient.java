package com.metawebthree.inventoryalert.infrastructure.client;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.supplychain.generated.rpc.*;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class InventoryClient {

    @DubboReference(check = false, lazy = true)
    private InventoryService inventoryService;

    /**
     * Get inventory details for a product
     * @param productId product ID
     * @param warehouseCode warehouse code
     * @return GetInventoryResponse
     */
    public GetInventoryResponse getInventory(Long productId, String warehouseCode) {
        GetInventoryRequest request = GetInventoryRequest.newBuilder()
                .setProductId(productId)
                .setWarehouseCode(warehouseCode)
                .build();
        
        try {
            return inventoryService.getInventory(request);
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to query inventory: " + e.getMessage());
        }
    }

    /**
     * Get all warehouse inventory for a product
     * @param productId product ID
     * @return list of inventory responses
     */
    public List<GetInventoryResponse> getAllWarehouseInventory(Long productId) {
        // Common warehouse codes
        String[] warehouses = {"DEFAULT", "MAIN", "BACKUP"};
        return java.util.Arrays.stream(warehouses)
                .map(w -> {
                    try {
                        return getInventory(productId, w);
                    } catch (Exception e) {
                        return null;
                    }
                })
                .filter(r -> r != null)
                .collect(java.util.stream.Collectors.toList());
    }

    /**
     * Check if stock is below threshold
     * @param productId product ID
     * @param threshold threshold value
     * @return true if below threshold
     */
    public boolean isBelowThreshold(Long productId, Integer threshold) {
        List<GetInventoryResponse> inventories = getAllWarehouseInventory(productId);
        int totalAvailable = inventories.stream()
                .mapToInt(GetInventoryResponse::getAvailableQty)
                .sum();
        return totalAvailable < threshold;
    }
}