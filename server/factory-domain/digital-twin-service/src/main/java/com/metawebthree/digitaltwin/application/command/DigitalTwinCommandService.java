package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
public class DigitalTwinCommandService {

    private final DigitalTwinDomainService domainService;
    private final DigitalTwinEventPublisher eventPublisher;
    private final DigitalTwinWebSocketHandler webSocketHandler;

    public DigitalTwinCommandService(
            DigitalTwinDomainService domainService,
            DigitalTwinEventPublisher eventPublisher,
            DigitalTwinWebSocketHandler webSocketHandler) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
        this.webSocketHandler = webSocketHandler;
    }

    // Device commands
    public Long registerDevice(String deviceCode, String deviceName, String deviceType,
                               String workshopId, String productionLineId) {
        Device device = domainService.registerDevice(
            deviceCode, deviceName, deviceType, workshopId, productionLineId);
        eventPublisher.publishDeviceRegistered(device.getDeviceCode());
        return device.getId();
    }

    public void updateDeviceStatus(String deviceCode, Device.DeviceStatus status) {
        domainService.updateDeviceStatus(deviceCode, status);
        eventPublisher.publishDeviceStatusChanged(deviceCode, status.name());
        webSocketHandler.broadcast(Map.of(
            "type", "DEVICE_STATUS_CHANGED",
            "data", Map.of(
                "eventType", "DEVICE_STATUS_CHANGED",
                "deviceCode", deviceCode,
                "status", status.name(),
                "timestamp", System.currentTimeMillis()
            )
        ));
    }

    public void deviceHeartbeat(String deviceCode) {
        domainService.deviceHeartbeat(deviceCode);
    }

    public void updateDevicePosition(String deviceCode, Double x, Double y, Double z, Double rotation) {
        domainService.updateDevicePosition(deviceCode, x, y, z, rotation);
        eventPublisher.publishDevicePositionUpdated(deviceCode, x, y, z);
        webSocketHandler.broadcast(Map.of(
            "type", "DEVICE_POSITION_UPDATED",
            "data", Map.of(
                "eventType", "DEVICE_POSITION_UPDATED",
                "deviceCode", deviceCode,
                "position", Map.of("x", x, "y", y, "z", z),
                "rotation", rotation,
                "timestamp", System.currentTimeMillis()
            )
        ));
    }

    // Workshop commands
    public Long createWorkshop(String workshopCode, String workshopName, String description) {
        var workshop = domainService.createWorkshop(workshopCode, workshopName, description);
        eventPublisher.publishWorkshopCreated(workshop.getWorkshopCode());
        return workshop.getId();
    }

    // ProductionLine commands
    public Long createProductionLine(String lineCode, String lineName, 
                                     String workshopId, Integer capacity) {
        var line = domainService.createProductionLine(lineCode, lineName, workshopId, capacity);
        eventPublisher.publishProductionLineCreated(line.getLineCode());
        return line.getId();
    }

    public void updateProductionLineOutput(String lineCode, Integer output) {
        domainService.updateProductionLineOutput(lineCode, output);
        eventPublisher.publishProductionOutputUpdated(lineCode, output);
        webSocketHandler.broadcast(Map.of(
            "type", "PRODUCTION_OUTPUT_UPDATED",
            "data", Map.of(
                "eventType", "PRODUCTION_OUTPUT_UPDATED",
                "lineCode", lineCode,
                "output", output,
                "timestamp", System.currentTimeMillis()
            )
        ));
    }

    // Alert commands
    public Long createAlert(String deviceCode, String workshopId, Alert.AlertLevel level,
                           Alert.AlertType type, String title, String description) {
        Alert alert = domainService.createAlert(deviceCode, workshopId, level, type, title, description);
        eventPublisher.publishAlertCreated(alert.getAlertCode(), level.name());
        webSocketHandler.broadcast(Map.of(
            "type", "ALERT_CREATED",
            "data", Map.of(
                "eventType", "ALERT_CREATED",
                "alertId", alert.getId(),
                "alertCode", alert.getAlertCode(),
                "deviceCode", deviceCode,
                "level", level.name(),
                "timestamp", System.currentTimeMillis()
            )
        ));
        return alert.getId();
    }

    public void acknowledgeAlert(Long alertId, String acknowledgedBy) {
        domainService.acknowledgeAlert(alertId, acknowledgedBy);
        eventPublisher.publishAlertAcknowledged(alertId);
    }

    public void resolveAlert(Long alertId, String solution, String resolvedBy) {
        domainService.resolveAlert(alertId, solution, resolvedBy);
        eventPublisher.publishAlertResolved(alertId);
    }
}