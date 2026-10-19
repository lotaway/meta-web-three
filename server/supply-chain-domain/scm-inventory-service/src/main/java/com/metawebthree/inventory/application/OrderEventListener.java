package com.metawebthree.inventory.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Event listener for order domain events.
 * Listens to order.created events and triggers inventory reservation.
 *
 * <p>Events are delivered at least once, so the listener de-duplicates on the
 * event id before doing any work. Reservations are keyed by the order id, the
 * same business key the order service uses for its synchronous reservation,
 * so re-processing can never consume inventory twice.</p>
 */
@Slf4j
@Component
public class OrderEventListener {

    static final String CONSUMER_GROUP = "inventory-service";

    private final InventoryApplicationService inventoryService;
    private final ProcessedEventStore processedEventStore;
    private final ObjectMapper objectMapper;

    public OrderEventListener(InventoryApplicationService inventoryService,
                               ProcessedEventStore processedEventStore,
                               ObjectMapper objectMapper) {
        this.inventoryService = inventoryService;
        this.processedEventStore = processedEventStore;
        this.objectMapper = objectMapper;
    }

    /**
     * Listen to order created events and reserve inventory.
     */
    @KafkaListener(topics = "order.created", groupId = CONSUMER_GROUP)
    public void onOrderCreated(String message) {
        try {
            Map<String, Object> event = objectMapper.readValue(message, Map.class);
            String eventId = (String) event.get("eventId");
            if (!processedEventStore.markFirstProcessing(eventId, CONSUMER_GROUP)) {
                log.info("Ignoring already processed order.created event: eventId={}", eventId);
                return;
            }

            String orderId = (String) event.get("orderId");
            List<Map<String, Object>> items = (List<Map<String, Object>>) event.get("items");

            log.info("Received order created event: orderId={}, itemCount={}",
                    orderId, items != null ? items.size() : 0);

            if (items != null) {
                for (Map<String, Object> item : items) {
                    String productId = (String) item.get("productId");
                    Integer quantity = (Integer) item.get("quantity");

                    // Reserve inventory for each item. The business id is the order id
                    // so this is idempotent with the order service's own reservation.
                    InventoryOperationResult result = inventoryService.reserveInventory(
                            Long.parseLong(productId),
                            quantity,
                            orderId
                    );

                    if (result != null && result.isSuccess()) {
                        log.info("Inventory reserved: productId={}, quantity={}, orderId={}",
                                productId, quantity, orderId);
                    } else {
                        log.warn("Failed to reserve inventory: productId={}, quantity={}, reason={}",
                                productId, quantity, result != null ? result.getMessage() : "unknown");
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to process order created event: {}", message, e);
        }
    }
}
