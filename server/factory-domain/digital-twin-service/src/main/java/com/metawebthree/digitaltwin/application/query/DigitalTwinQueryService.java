package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class DigitalTwinQueryService {

    private final DeviceRepository deviceRepository;
    private final WorkshopRepository workshopRepository;
    private final ProductionLineRepository productionLineRepository;
    private final AlertRepository alertRepository;
    private final DigitalTwinDomainService domainService;

    public DigitalTwinQueryService(
            DeviceRepository deviceRepository,
            WorkshopRepository workshopRepository,
            ProductionLineRepository productionLineRepository,
            AlertRepository alertRepository,
            DigitalTwinDomainService domainService) {
        this.deviceRepository = deviceRepository;
        this.workshopRepository = workshopRepository;
        this.productionLineRepository = productionLineRepository;
        this.alertRepository = alertRepository;
        this.domainService = domainService;
    }

    // Device queries
    public Optional<Device> getDeviceById(Long id) {
        return deviceRepository.findById(id);
    }

    public Optional<Device> getDeviceByCode(String deviceCode) {
        return deviceRepository.findByDeviceCode(deviceCode);
    }

    public List<Device> getAllDevices() {
        return deviceRepository.findAll();
    }

    public List<Device> getWorkshopDevices(String workshopId) {
        return deviceRepository.findByWorkshopId(workshopId);
    }

    public List<Device> getOnlineDevices() {
        return domainService.getOnlineDevices();
    }

    // Workshop queries
    public Optional<Workshop> getWorkshopById(Long id) {
        return workshopRepository.findById(id);
    }

    public List<Workshop> getAllWorkshops() {
        return workshopRepository.findAll();
    }

    // ProductionLine queries
    public Optional<ProductionLine> getProductionLineById(Long id) {
        return productionLineRepository.findById(id);
    }

    public List<ProductionLine> getAllProductionLines() {
        return productionLineRepository.findAll();
    }

    public List<ProductionLine> getWorkshopProductionLines(String workshopId) {
        return productionLineRepository.findByWorkshopId(workshopId);
    }

    // Alert queries
    public Optional<Alert> getAlertById(Long id) {
        return alertRepository.findById(id);
    }

    public List<Alert> getAllAlerts() {
        return alertRepository.findAll();
    }

    public List<Alert> getActiveAlerts() {
        return domainService.getActiveAlerts();
    }

    public List<Alert> getDeviceAlerts(String deviceCode) {
        return domainService.getDeviceAlerts(deviceCode);
    }

    public List<Alert> getWorkshopAlerts(String workshopId) {
        return alertRepository.findByWorkshopId(workshopId);
    }

    // Statistics
    public Long getOnlineDeviceCount() {
        return domainService.getOnlineDeviceCount();
    }

    public Long getActiveAlertCount() {
        return domainService.getActiveAlertCount();
    }

    public Double getAverageEfficiency() {
        return domainService.getAverageEfficiency();
    }
}