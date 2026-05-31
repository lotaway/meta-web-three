package com.metawebthree.order.infrastructure.client;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
@Slf4j
public class InventoryClient {

    @Value("${inventory.service.url:http://localhost:8082}")
    private String inventoryServiceUrl;
    
    private final RestTemplate restTemplate;

    public InventoryClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * Reserve inventory for order
     * @param orderId order ID
     * @param skuIds SKU ID list
     * @param quantities quantity list
     * @param reason reservation reason
     * @return true if success
     */
    public boolean reserveInventory(Long orderId, List<Long> skuIds, List<Integer> quantities, String reason) {
        try {
            if (skuIds == null || skuIds.isEmpty()) {
                log.warn("No SKUs to reserve for order: {}", orderId);
                return true;
            }
            
            // Use first SKU for now (simplified)
            Map<String, Object> request = new HashMap<>();
            request.put("skuCode", skuIds.get(0).toString());
            request.put("warehouseId", 1L);  // Default warehouse
            request.put("quantity", quantities != null && !quantities.isEmpty() ? quantities.get(0) : 1);
            request.put("bizId", orderId.toString());
            request.put("bizType", "ORDER");
            request.put("remark", reason);

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restTemplate.postForObject(
                inventoryServiceUrl + "/api/inventory/reserve",
                request,
                Map.class);

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Inventory reserved successfully - orderId: {}", orderId);
                return true;
            } else {
                log.warn("Inventory reservation failed - orderId: {}, response: {}", orderId, response);
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory reservation exception - orderId: {}, error: {}", orderId, e.getMessage());
            return false;
        }
    }

    /**
     * Confirm inventory reservation (deduct from reserved)
     * @param orderId order ID
     * @param reason confirmation reason
     * @return true if success
     */
    public boolean confirmInventoryReservation(Long orderId, String reason) {
        try {
            // Use bizId = orderId to confirm the reservation
            Map<String, Object> response = restTemplate.exchange(
                inventoryServiceUrl + "/api/inventory/confirm?bizId=" + orderId,
                HttpMethod.POST,
                HttpEntity.EMPTY,
                new ParameterizedTypeReference<Map<String, Object>>() {}).getBody();

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Inventory confirmation successful - orderId: {}", orderId);
                return true;
            } else {
                log.warn("Inventory confirmation failed - orderId: {}, response: {}", orderId, response);
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory confirmation exception - orderId: {}, error: {}", orderId, e.getMessage());
            return false;
        }
    }

    /**
     * Release inventory by order ID
     * @param orderId order ID
     * @param reason release reason
     * @return true if success
     */
    public boolean releaseInventoryByOrderId(Long orderId, String reason) {
        try {
            Map<String, Object> response = restTemplate.exchange(
                inventoryServiceUrl + "/api/inventory/cancel?bizId=" + orderId,
                HttpMethod.POST,
                HttpEntity.EMPTY,
                new ParameterizedTypeReference<Map<String, Object>>() {}).getBody();

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Inventory released by order ID successfully - orderId: {}", orderId);
                return true;
            } else {
                log.warn("Inventory release by order ID failed - orderId: {}, response: {}", orderId, response);
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory release by order ID exception - orderId: {}, error: {}", orderId, e.getMessage());
            return false;
        }
    }
}