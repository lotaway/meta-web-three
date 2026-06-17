package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.supplychain.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class InventoryClient {

    @DubboReference(check = false, lazy = true)
    private InventoryService inventoryService;

    @DubboReference(check = false, lazy = true)
    private InventoryAlertService inventoryAlertService;

    public Map<String, Object> getInventoryByProductId(String productId) {
        try {
            GetInventoryRequest request = GetInventoryRequest.newBuilder()
                    .setProductId(Long.parseLong(productId))
                    .build();
            GetInventoryResponse response = inventoryService.getInventory(request);
            return buildInventoryResult(response);
        } catch (Exception e) {
            log.error("Failed to get inventory by productId: {}, error: {}", productId, e.getMessage());
        }
        return new HashMap<>();
    }

    private Map<String, Object> buildInventoryResult(GetInventoryResponse response) {
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
    }

    public List<Map<String, Object>> getInventoryAlerts() {
        try {
            GetLowStockAlertsCountRequest countRequest = GetLowStockAlertsCountRequest.newBuilder().build();
            GetLowStockAlertsCountResponse countResponse = inventoryAlertService.getLowStockAlertsCount(countRequest);

            GetAlertStatisticsRequest statsRequest = GetAlertStatisticsRequest.newBuilder().build();
            GetAlertStatisticsResponse statsResponse = inventoryAlertService.getAlertStatistics(statsRequest);

            List<Map<String, Object>> alerts = new ArrayList<>();
            Map<String, Object> summary = new HashMap<>();
            summary.put("totalLowStockAlerts", countResponse.getCount());
            summary.put("statusDistribution", statsResponse.getDistributionMap());
            alerts.add(summary);
            return alerts;
        } catch (Exception e) {
            log.error("Failed to get inventory alerts: {}", e.getMessage());
        }
        return new ArrayList<>();
    }
}
