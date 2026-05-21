package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.springframework.stereotype.Service;

@Service
public class DigitalTwinCommandService {

    private final DigitalTwinDomainService domainService;
    private final DigitalTwinEventPublisher eventPublisher;

    public DigitalTwinCommandService(
            DigitalTwinDomainService domainService,
            DigitalTwinEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
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
    }

    public void deviceHeartbeat(String deviceCode) {
        domainService.deviceHeartbeat(deviceCode);
    }

    public void updateDevicePosition(String deviceCode, Double x, Double y, Double z, Double rotation) {
        domainService.updateDevicePosition(deviceCode, x, y, z, rotation);
        eventPublisher.publishDevicePositionUpdated(deviceCode, x, y, z);
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
    }

    // Alert commands
    public Long createAlert(String deviceCode, String workshopId, Alert.AlertLevel level,
                           Alert.AlertType type, String title, String description) {
        Alert alert = domainService.createAlert(deviceCode, workshopId, level, type, title, description);
        eventPublisher.publishAlertCreated(alert.getAlertCode(), level.name());
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