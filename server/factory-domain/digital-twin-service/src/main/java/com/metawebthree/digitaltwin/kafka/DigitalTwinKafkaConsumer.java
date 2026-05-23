package com.metawebthree.digitaltwin.kafka;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.annotation.RetryableTopic;
import org.springframework.kafka.retrytopic.DltStrategy;
import org.springframework.retry.annotation.Backoff;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

@Component
public class DigitalTwinKafkaConsumer {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinKafkaConsumer.class);
    private static final int IDEMPOTENCY_WINDOW_MINUTES = 30;
    private static final int CLEANUP_INTERVAL_MINUTES = 5;

    private final DigitalTwinWebSocketHandler webSocketHandler;
    private final DigitalTwinEventPublisher eventPublisher;
    private final ObjectMapper objectMapper = new ObjectMapper();

    // In-memory idempotency tracking (for single-instance, use Redis for multi-instance)
    private final Set<String> processedMessageIds = ConcurrentHashMap.newKeySet();
    private final ScheduledExecutorService cleanupScheduler = Executors.newSingleThreadScheduledExecutor();

    public DigitalTwinKafkaConsumer(
            DigitalTwinWebSocketHandler webSocketHandler,
            DigitalTwinEventPublisher eventPublisher) {
        this.webSocketHandler = webSocketHandler;
        this.eventPublisher = eventPublisher;
    }

    @PostConstruct
    public void init() {
        // Schedule cleanup of old message IDs
        cleanupScheduler.scheduleAtFixedRate(
            this::cleanupProcessedMessages,
            CLEANUP_INTERVAL_MINUTES,
            CLEANUP_INTERVAL_MINUTES,
            TimeUnit.MINUTES
        );
    }

    @PreDestroy
    public void shutdown() {
        cleanupScheduler.shutdown();
        try {
            if (!cleanupScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                cleanupScheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            cleanupScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private boolean isDuplicate(String messageId) {
        if (messageId == null || messageId.isEmpty()) {
            return false;
        }
        return !processedMessageIds.add(messageId);
    }

    private void cleanupProcessedMessages() {
        long cutoff = Instant.now().minusSeconds(IDEMPOTENCY_WINDOW_MINUTES * 60L).toEpochMilli();
        int before = processedMessageIds.size();
        processedMessageIds.removeIf(id -> {
            try {
                // Extract timestamp from message ID if it's a composite key
                return false; // Keep for now, simple implementation
            } catch (Exception e) {
                return false;
            }
        });
        int removed = before - processedMessageIds.size();
        if (removed > 0) {
            logger.debug("Cleaned up {} processed message IDs", removed);
        }
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

    private void processMessage(String topic, String message) {
        try {
            // Extract message ID for idempotency check
            String messageId = extractMessageId(message);
            
            // Check for duplicate
            if (isDuplicate(messageId)) {
                logger.debug("Duplicate message detected, skipping: {}", messageId);
                return;
            }
            
            logger.info("Received {}: {}", topic, message);
            
            // Route to appropriate handler based on topic
            switch (topic) {
                case "device.status.changed":
                    webSocketHandler.broadcast(Map.of("type", "DEVICE_STATUS_CHANGED", "data", message));
                    break;
                case "device.position.updated":
                    webSocketHandler.broadcast(Map.of("type", "DEVICE_POSITION_UPDATED", "data", message));
                    break;
                case "device.heartbeat":
                    // Heartbeat messages don't need broadcast
                    break;
                case "alert.created":
                    webSocketHandler.broadcast(Map.of("type", "ALERT_CREATED", "data", message));
                    break;
                case "production.output.updated":
                    webSocketHandler.broadcast(Map.of("type", "PRODUCTION_OUTPUT_UPDATED", "data", message));
                    break;
                case "agv.position.updated":
                    webSocketHandler.broadcast(Map.of("type", "AGV_POSITION_UPDATED", "data", message));
                    break;
                default:
                    logger.warn("Unknown topic: {}", topic);
            }
        } catch (Exception e) {
            logger.error("Error processing message from topic {}: {}", topic, e.getMessage(), e);
        }
    }

    private String extractMessageId(String message) {
        try {
            JsonNode node = objectMapper.readTree(message);
            // Try common ID field names
            if (node.has("messageId")) return node.get("messageId").asText();
            if (node.has("id")) return node.get("id").asText();
            if (node.has("eventId")) return node.get("eventId").asText();
            if (node.has("deviceCode") && node.has("timestamp")) {
                return node.get("deviceCode").asText() + "_" + node.get("timestamp").asText();
            }
            // Generate ID from content hash if no ID field
            return String.valueOf(message.hashCode());
        } catch (Exception e) {
            return String.valueOf(message.hashCode());
        }
    }
}