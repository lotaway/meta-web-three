package com.metawebthree.dataanalysis.infrastructure.client;

import com.metawebthree.common.generated.rpc.GetAlertStatisticsResponse;
import com.metawebthree.common.generated.rpc.GetLowStockAlertsCountRequest;
import com.metawebthree.common.generated.rpc.GetLowStockAlertsCountResponse;
import com.metawebthree.common.generated.rpc.InventoryAlertService;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
public class InventoryAlertClient {

    @DubboReference(check = false, lazy = true)
    private InventoryAlertService inventoryAlertService;

    /**
     * Get pending (unresolved) inventory alerts count
     * @return count of low stock alerts
     */
    public Long getLowStockAlertsCount() {
        try {
            GetLowStockAlertsCountRequest request = GetLowStockAlertsCountRequest.getDefaultInstance();
            GetLowStockAlertsCountResponse response = inventoryAlertService.getLowStockAlertsCount(request);
            return response.getCount();
        } catch (Exception e) {
            log.error("Failed to get low stock alerts count via Dubbo", e);
            return 0L;
        }
    }

    /**
     * Get all pending alerts statistics
     * @return map of status -> count
     */
    public Map<String, Long> getAlertStatistics() {
        try {
            GetAlertStatisticsResponse response = inventoryAlertService.getAlertStatistics(
                    com.metawebthree.common.generated.rpc.GetAlertStatisticsRequest.getDefaultInstance()
            );
            return new HashMap<>(response.getDistributionMap());
        } catch (Exception e) {
            log.error("Failed to get alert statistics via Dubbo", e);
            return new HashMap<>();
        }
    }
}