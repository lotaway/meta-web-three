package com.metawebthree.dataanalysis.infrastructure.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * Client for calling inventory-alert-service via REST API
 */
@Slf4j
@Component
public class InventoryAlertClient {

    private final RestTemplate restTemplate;

    @Value(\"${services.inventory-alert-service.url:http://localhost:8085}\")
    private String inventoryAlertServiceUrl;

    public InventoryAlertClient() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Get pending (unresolved) inventory alerts count
     * @return count of low stock alerts
     */
    public Long getLowStockAlertsCount() {
        try {
            String url = inventoryAlertServiceUrl + \"/api/admin/inventory-alert/statistics/low-stock-count\";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getBody() != null && response.getBody().containsKey(\"data\")) {
                Object data = response.getBody().get(\"data\");
                if (data instanceof Number) {
                    return ((Number) data).longValue();
                }
            }
        } catch (Exception e) {
            log.warn(\"Failed to get low stock alerts count: {}\", e.getMessage());
        }
        return 0L;
    }

    /**
     * Get all pending alerts
     * @return list of pending alert counts by level
     */
    public Map<String, Long> getAlertStatistics() {
        try {
            String url = inventoryAlertServiceUrl + \"/api/admin/inventory-alert/statistics/status-distribution\";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getBody() != null && response.getBody().containsKey(\"data\")) {
                @SuppressWarnings(\"unchecked\")
                Map<String, Long> data = (Map<String, Long>) response.getBody().get(\"data\");
                return data != null ? data : new HashMap<>();
            }
        } catch (Exception e) {
            log.warn(\"Failed to get alert statistics: {}\", e.getMessage());
        }
        return new HashMap<>();
    }
}