package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.UUID;

@Service
public class DigitalTwinDomainServiceImpl implements DigitalTwinDomainService {

    private final DeviceRepository deviceRepository;
    private final WorkshopRepository workshopRepository;
    private final ProductionLineRepository productionLineRepository;
    private final AlertRepository alertRepository;

    public DigitalTwinDomainServiceImpl(
            DeviceRepository deviceRepository,
            WorkshopRepository workshopRepository,
            ProductionLineRepository productionLineRepository,
            AlertRepository alertRepository) {
        this.deviceRepository = deviceRepository;
        this.workshopRepository = workshopRepository;
        this.productionLineRepository = productionLineRepository;
        this.alertRepository = alertRepository;
    }

    @Override
    public Device registerDevice(String deviceCode, String deviceName, String deviceType,
            String workshopId, String productionLineId) {
        Device device = new Device();
        device.create(deviceCode, deviceName, deviceType, workshopId, productionLineId);
        return deviceRepository.save(device);
    }

    @Override
    public void updateDeviceStatus(String deviceCode, Device.DeviceStatus status) {
        Device device = deviceRepository.findByDeviceCode(deviceCode)
            .orElseThrow(() -> new IllegalArgumentException("Device not found"));
        switch (status) {
            case ONLINE -> device.goOnline();
            case OFFLINE -> device.goOffline();
            case RUNNING -> device.startRunning();
            case IDLE -> device.setIdle();
            case ERROR -> device.reportError();
            case WARNING -> device.setWarning();
            case MAINTENANCE -> device.enterMaintenance();
        }
        deviceRepository.update(device);
    }

    @Override
    public void deviceHeartbeat(String deviceCode) {
        Device device = deviceRepository.findByDeviceCode(deviceCode)
            .orElseThrow(() -> new IllegalArgumentException("Device not found"));
        device.heartbeat();
        deviceRepository.update(device);
    }

    @Override
    public void updateDevicePosition(String deviceCode, Double x, Double y, Double z, Double rotation) {
        Device device = deviceRepository.findByDeviceCode(deviceCode)
            .orElseThrow(() -> new IllegalArgumentException("Device not found"));
        device.updatePosition(x, y, z, rotation);
        deviceRepository.update(device);
    }

    @Override
    public List<Device> getWorkshopDevices(String workshopId) {
        return deviceRepository.findByWorkshopId(workshopId);
    }

    @Override
    public List<Device> getOnlineDevices() {
        return deviceRepository.findByStatus(Device.DeviceStatus.ONLINE);
    }

    @Override
    public Workshop createWorkshop(String workshopCode, String workshopName, String description) {
        Workshop workshop = new Workshop();
        workshop.create(workshopCode, workshopName, description);
        return workshopRepository.save(workshop);
    }

    @Override
    public void updateWorkshopStatus(String workshopId, Workshop.WorkshopStatus status) {
        Workshop workshop = workshopRepository.findByWorkshopCode(workshopId)
            .orElseThrow(() -> new IllegalArgumentException("Workshop not found"));
        switch (status) {
            case CONSTRUCTION -> workshop.startConstruction();
            case OPERATING -> workshop.startOperating();
            case MAINTENANCE -> workshop.enterMaintenance();
            case DECOMMISSIONED -> workshop.decommission();
            default -> {}
        }
        workshopRepository.update(workshop);
    }

    @Override
    public List<Workshop> getAllWorkshops() {
        return workshopRepository.findAll();
    }

    @Override
    public ProductionLine createProductionLine(String lineCode, String lineName,
            String workshopId, Integer capacity) {
        ProductionLine line = new ProductionLine();
        line.create(lineCode, lineName, workshopId, capacity);
        return productionLineRepository.save(line);
    }

    @Override
    public void updateProductionLineStatus(String lineCode, ProductionLine.ProductionLineStatus status) {
        ProductionLine line = productionLineRepository.findByLineCode(lineCode)
            .orElseThrow(() -> new IllegalArgumentException("Production line not found"));
        switch (status) {
            case RUNNING -> line.start();
            case PAUSED -> line.pause();
            case IDLE -> line.stop();
            case MAINTENANCE -> line.maintenance();
            case BROKEN_DOWN -> line.breakdown();
        }
        productionLineRepository.update(line);
    }

    @Override
    public void updateProductionLineOutput(String lineCode, Integer output) {
        ProductionLine line = productionLineRepository.findByLineCode(lineCode)
            .orElseThrow(() -> new IllegalArgumentException("Production line not found"));
        line.updateOutput(output);
        productionLineRepository.update(line);
    }

    @Override
    public List<ProductionLine> getWorkshopProductionLines(String workshopId) {
        return productionLineRepository.findByWorkshopId(workshopId);
    }

    @Override
    public Alert createAlert(String deviceCode, String workshopId, Alert.AlertLevel level,
            Alert.AlertType type, String title, String description) {
        Alert alert = new Alert();
        String alertCode = "ALT-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        alert.create(alertCode, deviceCode, workshopId, level, type, title, description);
        return alertRepository.save(alert);
    }

    @Override
    public void acknowledgeAlert(Long alertId, String acknowledgedBy) {
        Alert alert = alertRepository.findById(alertId)
            .orElseThrow(() -> new IllegalArgumentException("Alert not found"));
        alert.acknowledge(acknowledgedBy);
        alertRepository.update(alert);
    }

    @Override
    public void resolveAlert(Long alertId, String solution, String resolvedBy) {
        Alert alert = alertRepository.findById(alertId)
            .orElseThrow(() -> new IllegalArgumentException("Alert not found"));
        alert.resolve(solution, resolvedBy);
        alertRepository.update(alert);
    }

    @Override
    public List<Alert> getActiveAlerts() {
        return alertRepository.findByStatus(Alert.AlertStatus.TRIGGERED);
    }

    @Override
    public List<Alert> getDeviceAlerts(String deviceCode) {
        return alertRepository.findByDeviceCode(deviceCode);
    }

    @Override
    public Long getOnlineDeviceCount() {
        return (long) deviceRepository.findByStatus(Device.DeviceStatus.ONLINE).size();
    }

    @Override
    public Long getActiveAlertCount() {
        return (long) alertRepository.findByStatus(Alert.AlertStatus.TRIGGERED).size();
    }

    @Override
    public Double getAverageEfficiency() {
        List<ProductionLine> lines = productionLineRepository.findAll();
        if (lines.isEmpty()) return 0.0;
        double total = lines.stream().mapToDouble(ProductionLine::getEfficiency).sum();
        return total / lines.size();
    }
}