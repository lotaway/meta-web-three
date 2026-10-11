package com.metawebthree.digitaltwin.kafka;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.digitaltwin.application.ai.WarehouseAIService;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.annotation.RetryableTopic;
import org.springframework.kafka.retrytopic.DltStrategy;
import org.springframework.retry.annotation.Backoff;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class DigitalTwinKafkaConsumer {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinKafkaConsumer.class);
    private static final int DEFAULT_ANOMALY_DETECTION_HOURS = 24;
    private static final double DEFAULT_CONFIDENCE = 0.0;
    private static final String MESSAGE_ID_DELIMITER = "_";

    private final DigitalTwinWebSocketHandler webSocketHandler;
    private final WarehouseAIService aiService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    // Kafka consumer group provides built-in deduplication via offset commits
    // No additional in-memory state required - relying on at-least-once delivery semantics

    public DigitalTwinKafkaConsumer(
            DigitalTwinWebSocketHandler webSocketHandler,
            WarehouseAIService aiService) {
        this.webSocketHandler = webSocketHandler;
        this.aiService = aiService;
    }

    private boolean isDuplicate(String messageId) {
        // Rely on Kafka consumer group offset commits for deduplication
        return messageId == null || messageId.isEmpty();
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR,
        include = {Exception.class}
    )
    @KafkaListener(topics = "device.status.changed", groupId = "digital-twin")
    public void consumeDeviceStatusChanged(String message) {
        processMessage("device.status.changed", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "device.position.updated", groupId = "digital-twin")
    public void consumeDevicePositionUpdated(String message) {
        processMessage("device.position.updated", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "device.heartbeat", groupId = "digital-twin")
    public void consumeDeviceHeartbeat(String message) {
        processMessage("device.heartbeat", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "alert.created", groupId = "digital-twin")
    public void consumeAlertCreated(String message) {
        processMessage("alert.created", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "production.output.updated", groupId = "digital-twin")
    public void consumeProductionOutputUpdated(String message) {
        processMessage("production.output.updated", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "agv.position.updated", groupId = "digital-twin")
    public void consumeAgvPositionUpdated(String message) {
        processMessage("agv.position.updated", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "warehouse.status.changed", groupId = "digital-twin")
    public void consumeWarehouseStatusChanged(String message) {
        processMessage("warehouse.status.changed", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "inventory.level.changed", groupId = "digital-twin")
    public void consumeInventoryLevelChanged(String message) {
        processMessage("inventory.level.changed", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "inventory.alert.created", groupId = "digital-twin")
    public void consumeInventoryAlertCreated(String message) {
        processMessage("inventory.alert.created", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "restock.suggestion.created", groupId = "digital-twin")
    public void consumeRestockSuggestionCreated(String message) {
        processMessage("restock.suggestion.created", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "shelf.status.changed", groupId = "digital-twin")
    public void consumeShelfStatusChanged(String message) {
        processMessage("shelf.status.changed", message);
    }

    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaListener(topics = "forecast.computed", groupId = "digital-twin")
    public void consumeForecastComputed(String message) {
        processMessage("forecast.computed", message);
    }

    private void processMessage(String topic, String message) {
        try {
            String messageId = extractMessageId(message);
            if (isDuplicate(messageId)) {
                logger.info("Duplicate message detected, skipping: {}", messageId);
                return;
            }
            logger.info("Received {}: {}", topic, message);
            dispatchToHandler(topic, message);
        } catch (Exception e) {
            logger.error("Error processing message from topic {}: {}", topic, message, e);
        }
    }

    private void dispatchToHandler(String topic, String message) {
        switch (topic) {
            case "device.status.changed" -> handleDeviceStatusChanged(message);
            case "device.position.updated" -> handleDevicePositionUpdated(message);
            case "device.heartbeat" -> handleDeviceHeartbeat(message);
            case "alert.created" -> handleAlertCreated(message);
            case "production.output.updated" -> handleProductionOutputUpdated(message);
            case "agv.position.updated" -> handleAgvPositionUpdated(message);
            case "warehouse.status.changed" -> handleWarehouseStatusChanged(message);
            case "inventory.level.changed" -> handleInventoryLevelChanged(message);
            case "inventory.alert.created" -> handleInventoryAlertCreated(message);
            case "restock.suggestion.created" -> handleRestockSuggestionCreated(message);
            case "shelf.status.changed" -> handleShelfStatusChanged(message);
            case "forecast.computed" -> handleForecastComputed(message);
            default -> logger.warn("Unknown topic: {}", topic);
        }
    }

    private void handleDeviceStatusChanged(String message) {
        webSocketHandler.broadcast(Map.of("type", "DEVICE_STATUS_CHANGED", "data", message));
    }

    private void handleDevicePositionUpdated(String message) {
        webSocketHandler.broadcast(Map.of("type", "DEVICE_POSITION_UPDATED", "data", message));
    }

    private void handleDeviceHeartbeat(String message) {
        logger.info("Device heartbeat received, event ignored for digital twin: {}", message);
    }

    private void handleAlertCreated(String message) {
        webSocketHandler.broadcast(Map.of("type", "ALERT_CREATED", "data", message));
    }

    private void handleProductionOutputUpdated(String message) {
        webSocketHandler.broadcast(Map.of("type", "PRODUCTION_OUTPUT_UPDATED", "data", message));
    }

    private void handleAgvPositionUpdated(String message) {
        webSocketHandler.broadcast(Map.of("type", "AGV_POSITION_UPDATED", "data", message));
    }

    private void handleWarehouseStatusChanged(String message) {
        webSocketHandler.broadcast(Map.of("type", "WAREHOUSE_STATUS_CHANGED", "data", message));
    }

    private void handleInventoryLevelChanged(String message) {
        webSocketHandler.broadcast(Map.of("type", "INVENTORY_LEVEL_CHANGED", "data", message));
        detectAnomalyForInventory(message);
    }

    private void detectAnomalyForInventory(String message) {
        try {
            JsonNode node = objectMapper.readTree(message);
            if (!node.has("sku") || !node.has("warehouseId") || node.get("sku").isNull()) {
                return;
            }
            String skuCode = node.get("sku").asText();
            Long warehouseId = node.get("warehouseId").asLong();
            invokeAndBroadcastAnomalies(skuCode, warehouseId);
        } catch (Exception e) {
            logger.error("Failed to trigger anomaly detection for message: {}", message, e);
        }
    }

    private void invokeAndBroadcastAnomalies(String skuCode, Long warehouseId) {
        logger.info("Triggering AI anomaly detection for SKU {} in warehouse {}", skuCode, warehouseId);
        var anomalies = aiService.detectAnomalies(skuCode, warehouseId, DEFAULT_ANOMALY_DETECTION_HOURS);
        if (anomalies != null && !anomalies.isEmpty()) {
            logger.info("Detected {} anomalies for SKU {} in warehouse {}", 
                anomalies.size(), skuCode, warehouseId);
            webSocketHandler.broadcast(Map.of(
                "type", "ANOMALY_DETECTED",
                "data", Map.of(
                    "skuCode", skuCode,
                    "warehouseId", warehouseId,
                    "anomalies", anomalies
                )
            ));
        }
    }

    private void handleForecastComputed(String message) {
        try {
            JsonNode node = objectMapper.readTree(message);
            String skuCode = node.has("skuCode") ? node.get("skuCode").asText() : null;
            Long warehouseId = node.has("warehouseId") ? node.get("warehouseId").asLong() : null;
            Integer predictedQty = node.has("predictedQuantity") ? node.get("predictedQuantity").asInt() : null;
            Double confidence = node.has("confidence") ? node.get("confidence").asDouble() : null;
            
            logger.info("Forecast computed for SKU {} in warehouse {}: predicted={}, confidence={}",
                skuCode, warehouseId, predictedQty, confidence);
            
            if (skuCode != null && warehouseId != null && predictedQty != null) {
                adjustSafetyStock(warehouseId, skuCode, predictedQty, confidence);
            }
            
            webSocketHandler.broadcast(Map.of("type", "FORECAST_COMPUTED", "data", message));
        } catch (Exception e) {
            logger.error("Failed to process forecast.computed message: {}", message, e);
        }
    }

    private void adjustSafetyStock(Long warehouseId, String skuCode, Integer predictedQty, Double confidence) {
        logger.info("Adjusting safety stock for warehouse {} SKU {} based on forecast: predicted={}",
            warehouseId, skuCode, predictedQty);
        
        webSocketHandler.broadcast(Map.of(
            "type", "SAFETY_STOCK_ADJUSTED",
            "data", Map.of(
                "warehouseId", warehouseId,
                "skuCode", skuCode,
                "predictedQuantity", predictedQty,
                "confidence", confidence != null ? confidence : DEFAULT_CONFIDENCE,
                "adjustedAt", System.currentTimeMillis()
            )
        ));
    }

    private void handleInventoryAlertCreated(String message) {
        webSocketHandler.broadcast(Map.of("type", "INVENTORY_ALERT_CREATED", "data", message));
    }

    private void handleRestockSuggestionCreated(String message) {
        webSocketHandler.broadcast(Map.of("type", "RESTOCK_SUGGESTION_CREATED", "data", message));
    }

    private void handleShelfStatusChanged(String message) {
        webSocketHandler.broadcast(Map.of("type", "SHELF_STATUS_CHANGED", "data", message));
    }

    private String extractMessageId(String message) {
        try {
            JsonNode node = objectMapper.readTree(message);
            if (node.has("messageId")) return node.get("messageId").asText();
            if (node.has("id")) return node.get("id").asText();
            if (node.has("eventId")) return node.get("eventId").asText();
            if (node.has("deviceCode") && node.has("timestamp")) {
                return node.get("deviceCode").asText() + MESSAGE_ID_DELIMITER + node.get("timestamp").asText();
            }
            return String.valueOf(message.hashCode());
        } catch (Exception e) {
            logger.warn("Failed to extract message ID, using hashCode fallback: {}", e.getMessage());
            return String.valueOf(message.hashCode());
        }
    }
}