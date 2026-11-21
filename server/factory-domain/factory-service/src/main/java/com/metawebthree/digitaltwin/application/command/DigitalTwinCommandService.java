package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.application.query.DigitalTwinQueryService;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPayload;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventType;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;

@Service
public class DigitalTwinCommandService {

    private final DigitalTwinDomainService domainService;
    private final DigitalTwinQueryService queryService;
    private final DigitalTwinEventPublisher eventPublisher;
    private final DigitalTwinWebSocketHandler webSocketHandler;

    public DigitalTwinCommandService(
            DigitalTwinDomainService domainService,
            DigitalTwinQueryService queryService,
            DigitalTwinEventPublisher eventPublisher,
            DigitalTwinWebSocketHandler webSocketHandler) {
        this.domainService = domainService;
        this.queryService = queryService;
        this.eventPublisher = eventPublisher;
        this.webSocketHandler = webSocketHandler;
    }

    private void broadcast(DigitalTwinEventType type, Map<String, Object> fields) {
        webSocketHandler.broadcast(Map.of("type", type.name(), "data", DigitalTwinEventPayload.of(type, fields)));
    }

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
        broadcast(DigitalTwinEventType.DEVICE_STATUS_CHANGED, Map.of(
            "deviceCode", deviceCode,
            "status", status.name()
        ));
    }

    public void deviceHeartbeat(String deviceCode) {
        domainService.deviceHeartbeat(deviceCode);
    }

    public void updateDevicePosition(String deviceCode, Double x, Double y, Double z, Double rotation) {
        domainService.updateDevicePosition(deviceCode, x, y, z, rotation);
        eventPublisher.publishDevicePositionUpdated(deviceCode, x, y, z);
        broadcast(DigitalTwinEventType.DEVICE_POSITION_UPDATED, Map.of(
            "deviceCode", deviceCode,
            "position", Map.of("x", x, "y", y, "z", z),
            "rotation", rotation
        ));
    }

    public Long createWorkshop(String workshopCode, String workshopName, String description) {
        var workshop = domainService.createWorkshop(workshopCode, workshopName, description);
        eventPublisher.publishWorkshopCreated(workshop.getWorkshopCode());
        return workshop.getId();
    }

    public Long createProductionLine(String lineCode, String lineName,
                                     String workshopId, Integer capacity) {
        var line = domainService.createProductionLine(lineCode, lineName, workshopId, capacity);
        eventPublisher.publishProductionLineCreated(line.getLineCode());
        return line.getId();
    }

    public void updateProductionLineOutput(String lineCode, Integer output) {
        domainService.updateProductionLineOutput(lineCode, output);
        eventPublisher.publishProductionOutputUpdated(lineCode, output);
        broadcast(DigitalTwinEventType.PRODUCTION_OUTPUT_UPDATED, Map.of(
            "lineCode", lineCode,
            "output", output,
            "stats", Map.of(
                "onlineDeviceCount", queryService.getOnlineDeviceCount(),
                "activeAlertCount", queryService.getActiveAlertCount(),
                "averageEfficiency", queryService.getAverageEfficiency()
            )
        ));
    }

    public Long createAlert(String deviceCode, String workshopId, Alert.AlertLevel level,
                            Alert.AlertType type, String title, String description) {
        Alert alert = domainService.createAlert(deviceCode, workshopId, level, type, title, description);
        eventPublisher.publishAlertCreated(alert.getAlertCode(), level.name());
        broadcast(DigitalTwinEventType.ALERT_CREATED, alertFields(alert));
        return alert.getId();
    }

    private Map<String, Object> alertFields(Alert alert) {
        Map<String, Object> fields = new HashMap<>();
        fields.put("alertId", alert.getId());
        fields.put("alertCode", alert.getAlertCode());
        fields.put("deviceCode", alert.getDeviceCode());
        fields.put("workshopId", alert.getWorkshopId());
        fields.put("level", alert.getLevel().name());
        fields.put("type", alert.getType().name());
        fields.put("title", alert.getTitle());
        fields.put("description", alert.getDescription());
        fields.put("status", alert.getStatus().name());
        fields.put("occurredAt", DigitalTwinEventPayload.format(alert.getOccurredAt()));
        fields.put("createdAt", DigitalTwinEventPayload.format(alert.getCreatedAt()));
        return fields;
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
