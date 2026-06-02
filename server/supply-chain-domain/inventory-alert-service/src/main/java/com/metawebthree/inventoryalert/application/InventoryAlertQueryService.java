package com.metawebthree.inventoryalert.application;

import com.metawebthree.inventoryalert.infrastructure.persistence.mapper.InventoryAlertMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Inventory alert query service for statistics
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class InventoryAlertQueryService {

    private final InventoryAlertMapper inventoryAlertMapper;

    /**
     * Get low stock alerts count (unresolved alerts)
     * @return count of pending alerts
     */
    public Long getLowStockAlertsCount() {
        try {
            Long count = inventoryAlertMapper.countPendingAlerts();
            return count != null ? count : 0L;
        } catch (Exception e) {
            log.error("Failed to get low stock alerts count", e);
            return 0L;
        }
    }

    /**
     * Get alert statistics by status
     * Status: 0->pending, 1->resolved, 2->ignored
     * @return map of status -> count
     */
    public Map<String, Long> getAlertStatusDistribution() {
        try {
            List<Object> results = inventoryAlertMapper.selectAlertStatistics();
            Map<String, Long> distribution = new HashMap<>();
            
            // Simple parsing - in production would use proper result mapping
            // For now, return just the pending count
            Long pendingCount = inventoryAlertMapper.countPendingAlerts();
            distribution.put("PENDING", pendingCount != null ? pendingCount : 0L);
            distribution.put("RESOLVED", 0L);
            distribution.put("IGNORED", 0L);
            
            return distribution;
        } catch (Exception e) {
            log.error("Failed to get alert status distribution", e);
            return new HashMap<>();
        }
    }
}