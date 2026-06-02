package com.metawebthree.inventoryalert.interfaces.admin;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.inventoryalert.application.InventoryAlertQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Admin inventory alert statistics controller
 * Provides alert statistics for real-time dashboard
 */
@RestController
@RequestMapping("/api/admin/inventory-alert/statistics")
@RequiredArgsConstructor
public class InventoryAlertStatisticsController {

    private final InventoryAlertQueryService inventoryAlertQueryService;

    /**
     * Get low stock alerts count (unresolved alerts)
     * @return count of pending alerts
     */
    @GetMapping("/low-stock-count")
    public ApiResponse<Long> getLowStockAlertsCount() {
        return ApiResponse.success(inventoryAlertQueryService.getLowStockAlertsCount());
    }

    /**
     * Get alert statistics by status
     * @return map of status -> count
     */
    @GetMapping("/status-distribution")
    public ApiResponse<Map<String, Long>> getAlertStatusDistribution() {
        return ApiResponse.success(inventoryAlertQueryService.getAlertStatusDistribution());
    }
}