package com.metawebthree.gateway.client;

import com.metawebthree.supplychain.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class InventoryClient {

    @DubboReference
    private InventoryService inventoryService;

    /**
     * Get inventory by product ID
     * @param productId product ID
     * @return inventory data map
     */
    public Map<String, Object> getInventoryByProductId(String productId) {
        try {
            GetInventoryRequest request = GetInventoryRequest.newBuilder()
                    .setProductId(Long.parseLong(productId))
                    .build();
            GetInventoryResponse response = inventoryService.getInventory(request);
            
            Map<String, Object> result = new HashMap<>();
            if (response.getProductId() > 0) {
                result.put("id", response.getProductId());
                result.put("productId", response.getProductId());
                result.put("quantity", response.getTotalQty());
                result.put("availableQuantity", response.getAvailableQty());
                result.put("reservedQuantity", response.getReservedQty());
                result.put("warehouseCode", response.getWarehouseCode());
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get inventory by productId: {}, error: {}", productId, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get inventory alerts - requires InventoryAlertService
     * @return list of inventory alerts
     */
    public List<Map<String, Object>> getInventoryAlerts() {
        log.warn("getInventoryAlerts via Dubbo not implemented - requires InventoryAlertService");
        return new ArrayList<>();
    }
}