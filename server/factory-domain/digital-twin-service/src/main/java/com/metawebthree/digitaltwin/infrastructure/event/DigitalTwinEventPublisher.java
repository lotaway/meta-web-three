package com.metawebthree.digitaltwin.infrastructure.event;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class DigitalTwinEventPublisher {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinEventPublisher.class);

    public void publishDeviceRegistered(String deviceCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_REGISTERED");
        event.put("deviceCode", deviceCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishDeviceStatusChanged(String deviceCode, String status) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_STATUS_CHANGED");
        event.put("deviceCode", deviceCode);
        event.put("status", status);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishDevicePositionUpdated(String deviceCode, Double x, Double y, Double z) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "DEVICE_POSITION_UPDATED");
        event.put("deviceCode", deviceCode);
        event.put("position", Map.of("x", x, "y", y, "z", z));
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishWorkshopCreated(String workshopCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORKSHOP_CREATED");
        event.put("workshopCode", workshopCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishProductionLineCreated(String lineCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "PRODUCTION_LINE_CREATED");
        event.put("lineCode", lineCode);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishProductionOutputUpdated(String lineCode, Integer output) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "PRODUCTION_OUTPUT_UPDATED");
        event.put("lineCode", lineCode);
        event.put("output", output);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishAlertCreated(String alertCode, String level) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_CREATED");
        event.put("alertCode", alertCode);
        event.put("level", level);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishAlertAcknowledged(Long alertId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_ACKNOWLEDGED");
        event.put("alertId", alertId);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }

    public void publishAlertResolved(Long alertId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "ALERT_RESOLVED");
        event.put("alertId", alertId);
        event.put("timestamp", System.currentTimeMillis());
        logger.info("Publishing: {}", event);
    }
}