package com.metawebthree.digitaltwin.kafka;

import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import java.util.Map;

@Component
public class DigitalTwinKafkaConsumer {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinKafkaConsumer.class);
    
    private final DigitalTwinWebSocketHandler webSocketHandler;
    private final DigitalTwinEventPublisher eventPublisher;

    public DigitalTwinKafkaConsumer(
            DigitalTwinWebSocketHandler webSocketHandler,
            DigitalTwinEventPublisher eventPublisher) {
        this.webSocketHandler = webSocketHandler;
        this.eventPublisher = eventPublisher;
    }

    @KafkaListener(topics = "device.status.changed", groupId = "digital-twin")
    public void consumeDeviceStatusChanged(String message) {
        logger.info("Received device status changed: {}", message);
        webSocketHandler.broadcast(Map.of("type", "DEVICE_STATUS_CHANGED", "data", message));
    }

    @KafkaListener(topics = "device.position.updated", groupId = "digital-twin")
    public void consumeDevicePositionUpdated(String message) {
        logger.info("Received device position updated: {}", message);
        webSocketHandler.broadcast(Map.of("type", "DEVICE_POSITION_UPDATED", "data", message));
    }

    @KafkaListener(topics = "device.heartbeat", groupId = "digital-twin")
    public void consumeDeviceHeartbeat(String message) {
        logger.info("Received device heartbeat: {}", message);
    }

    @KafkaListener(topics = "alert.created", groupId = "digital-twin")
    public void consumeAlertCreated(String message) {
        logger.info("Received alert created: {}", message);
        webSocketHandler.broadcast(Map.of("type", "ALERT_CREATED", "data", message));
    }

    @KafkaListener(topics = "production.output.updated", groupId = "digital-twin")
    public void consumeProductionOutputUpdated(String message) {
        logger.info("Received production output updated: {}", message);
        webSocketHandler.broadcast(Map.of("type", "PRODUCTION_OUTPUT_UPDATED", "data", message));
    }

    @KafkaListener(topics = "agv.position.updated", groupId = "digital-twin")
    public void consumeAgvPositionUpdated(String message) {
        logger.info("Received AGV position updated: {}", message);
        webSocketHandler.broadcast(Map.of("type", "AGV_POSITION_UPDATED", "data", message));
    }
}