package com.metawebthree.inventory.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

/**
 * Event listener for order domain events.
 * Listens to order.created events and triggers inventory reservation.
 */
@Slf4j
@Component
public class OrderEventListener {

    private final InventoryApplicationService inventoryService;
    private final ObjectMapper objectMapper;

    public OrderEventListener(InventoryApplicationService inventoryService,
                               ObjectMapper objectMapper) {
        this.inventoryService = inventoryService;
        this.objectMapper = objectMapper;
    }

    /**
     * Listen to order created events and reserve inventory.
     */
    @KafkaListener(topics = "order.created", groupId = "inventory-service")
    public void onOrderCreated(String message) {
        try {
            Map<String, Object> event = objectMapper.readValue(message, Map.class);
            String orderId = (String) event.get("orderId");
            List<Map<String, Object>> items = (List<Map<String, Object>>) event.get("items");

            log.info("Received order created event: orderId={}, itemCount={}",
                    orderId, items != null ? items.size() : 0);

            if (items != null) {
                for (Map<String, Object> item : items) {
                    String productId = (String) item.get("productId");
                    Integer quantity = (Integer) item.get("quantity");
                    // Convert unitPrice
                    Object priceObj = item.get("unitPrice");
                    BigDecimal unitPrice = new BigDecimal(priceObj.toString());

                    // Reserve inventory for each item
                    boolean success = inventoryService.reserveInventory(
                            Long.parseLong(productId),
                            quantity,
                            "ORDER-" + orderId
                    );

                    if (success) {
                        log.info("Inventory reserved: productId={}, quantity={}, orderId={}",
                                productId, quantity, orderId);
                    } else {
                        log.warn("Failed to reserve inventory: productId={}, quantity={}",
                                productId, quantity);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to process order created event: {}", message, e);
        }
    }
}