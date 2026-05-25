package com.metawebthree.digitaltwin.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Component
public class DigitalTwinEventPublisher {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinEventPublisher.class);
    
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${digital-twin.kafka.topic-prefix:digital-twin}")
    private String topicPrefix;

    public DigitalTwinEventPublisher(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = new ObjectMapper();
    }

    private void publishToKafka(String topic, Map<String, Object> event) {
        try {
            String json = objectMapper.writeValueAsString(event);
            CompletableFuture<SendResult<String, String>> future = kafkaTemplate.send(topic, json);
            
            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    logger.error("Failed to send event to Kafka. topic={}, event={}", topic, event, ex);
                } else {
                    logger.debug("Event sent to Kafka. topic={}, partition={}, offset={}", 
                        topic, result.getRecordMetadata().partition(), 
                        result.getRecordMetadata().offset());
                }
            });
        } catch (JsonProcessingException e) {
            logger.error("Failed to serialize event: {}", event, e);
        }
    }

    public void publishDeviceRegistered(String deviceCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_REGISTERED");
        event.put("deviceCode", deviceCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing DEVICE_REGISTERED: {}", deviceCode);
        publishToKafka(topicPrefix + ".device.registered", event);
    }

    public void publishDeviceStatusChanged(String deviceCode, String status) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_STATUS_CHANGED");
        event.put("deviceCode", deviceCode);
        event.put("status", status);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing DEVICE_STATUS_CHANGED: {} -> {}", deviceCode, status);
        publishToKafka(topicPrefix + ".device.status.changed", event);
    }

    public void publishDevicePositionUpdated(String deviceCode, Double x, Double y, Double z) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_POSITION_UPDATED");
        event.put("deviceCode", deviceCode);
        event.put("position", Map.of("x", x, "y", y, "z", z));
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing DEVICE_POSITION_UPDATED: {} at ({},{},{})", deviceCode, x, y, z);
        publishToKafka(topicPrefix + ".device.position.updated", event);
    }

    public void publishWorkshopCreated(String workshopCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORKSHOP_CREATED");
        event.put("workshopCode", workshopCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing WORKSHOP_CREATED: {}", workshopCode);
        publishToKafka(topicPrefix + ".workshop.created", event);
    }

    public void publishProductionLineCreated(String lineCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "PRODUCTION_LINE_CREATED");
        event.put("lineCode", lineCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing PRODUCTION_LINE_CREATED: {}", lineCode);
        publishToKafka(topicPrefix + ".production.line.created", event);
    }

    public void publishProductionOutputUpdated(String lineCode, Integer output) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "PRODUCTION_OUTPUT_UPDATED");
        event.put("lineCode", lineCode);
        event.put("output", output);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing PRODUCTION_OUTPUT_UPDATED: {} = {}", lineCode, output);
        publishToKafka(topicPrefix + ".production.output.updated", event);
    }

    public void publishAlertCreated(String alertCode, String level) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_CREATED");
        event.put("alertCode", alertCode);
        event.put("level", level);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing ALERT_CREATED: {} [{}]", alertCode, level);
        publishToKafka(topicPrefix + ".alert.created", event);
    }

    public void publishAlertAcknowledged(Long alertId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_ACKNOWLEDGED");
        event.put("alertId", alertId);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing ALERT_ACKNOWLEDGED: id={}", alertId);
        publishToKafka(topicPrefix + ".alert.acknowledged", event);
    }

    public void publishAlertResolved(Long alertId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_RESOLVED");
        event.put("alertId", alertId);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing ALERT_RESOLVED: id={}", alertId);
        publishToKafka(topicPrefix + ".alert.resolved", event);
    }

    public void publishWarehouseStatusChanged(String warehouseCode, String status) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WAREHOUSE_STATUS_CHANGED");
        event.put("warehouseCode", warehouseCode);
        event.put("status", status);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing WAREHOUSE_STATUS_CHANGED: {} -> {}", warehouseCode, status);
        publishToKafka(topicPrefix + ".warehouse.status.changed", event);
    }

    public void publishInventoryLevelChanged(String warehouseCode, String sku, Integer quantity, String status) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "INVENTORY_LEVEL_CHANGED");
        event.put("warehouseCode", warehouseCode);
        event.put("sku", sku);
        event.put("quantity", quantity);
        event.put("status", status);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing INVENTORY_LEVEL_CHANGED: {} SKU={} qty={}", warehouseCode, sku, quantity);
        publishToKafka(topicPrefix + ".inventory.level.changed", event);
    }

    public void publishInventoryAlertCreated(String warehouseCode, String alertCode, String level, String message) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "INVENTORY_ALERT_CREATED");
        event.put("warehouseCode", warehouseCode);
        event.put("alertCode", alertCode);
        event.put("level", level);
        event.put("message", message);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing INVENTORY_ALERT_CREATED: {} [{}]", alertCode, level);
        publishToKafka(topicPrefix + ".inventory.alert.created", event);
    }

    public void publishRestockSuggestionCreated(String warehouseCode, String sku, Integer suggestedQty, String reason) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RESTOCK_SUGGESTION_CREATED");
        event.put("warehouseCode", warehouseCode);
        event.put("sku", sku);
        event.put("suggestedQty", suggestedQty);
        event.put("reason", reason);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing RESTOCK_SUGGESTION_CREATED: SKU={} qty={}", sku, suggestedQty);
        publishToKafka(topicPrefix + ".restock.suggestion.created", event);
    }

    public void publishShelfStatusChanged(String warehouseCode, String shelfCode, String status) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "SHELF_STATUS_CHANGED");
        event.put("warehouseCode", warehouseCode);
        event.put("shelfCode", shelfCode);
        event.put("status", status);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing SHELF_STATUS_CHANGED: {} shelf={}", warehouseCode, shelfCode);
        publishToKafka(topicPrefix + ".shelf.status.changed", event);
    }
}