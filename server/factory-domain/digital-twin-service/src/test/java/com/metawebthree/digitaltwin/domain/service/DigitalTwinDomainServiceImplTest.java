package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinDomainServiceImplTest {

    @Mock
    private DeviceRepository deviceRepository;
    @Mock
    private WorkshopRepository workshopRepository;
    @Mock
    private ProductionLineRepository productionLineRepository;
    @Mock
    private AlertRepository alertRepository;

    private DigitalTwinDomainServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new DigitalTwinDomainServiceImpl(
            deviceRepository,
            workshopRepository,
            productionLineRepository,
            alertRepository
        );
    }

    @Test
    void registerDevice_shouldCreateDeviceSuccessfully() {
        Device savedDevice = new Device();
        savedDevice.setId(1L);
        savedDevice.setDeviceCode("DEV-001");
        savedDevice.setDeviceName("Test Device");
        
        when(deviceRepository.save(any(Device.class))).thenReturn(savedDevice);

        Device result = service.registerDevice(
            "DEV-001", "Test Device", "SENSOR", "WS001", "PL001"
        );

        assertNotNull(result);
        assertEquals("DEV-001", result.getDeviceCode());
        verify(deviceRepository).save(any(Device.class));
    }

    @Test
    void updateDeviceStatus_shouldUpdateToOnline() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        device.setStatus(Device.DeviceStatus.OFFLINE);
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));
        when(deviceRepository.update(any(Device.class))).thenReturn(device);

        service.updateDeviceStatus("DEV-001", Device.DeviceStatus.ONLINE);

        assertEquals(Device.DeviceStatus.ONLINE, device.getStatus());
        verify(deviceRepository).update(device);
    }

    @Test
    void updateDeviceStatus_shouldThrowWhenDeviceNotFound() {
        when(deviceRepository.findByDeviceCode("DEV-999")).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () ->
            service.updateDeviceStatus("DEV-999", Device.DeviceStatus.ONLINE)
        );
    }

    @Test
    void updateDeviceStatus_shouldUpdateToRunning() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        device.setStatus(Device.DeviceStatus.IDLE);
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));
        when(deviceRepository.update(any(Device.class))).thenReturn(device);

        service.updateDeviceStatus("DEV-001", Device.DeviceStatus.RUNNING);

        assertEquals(Device.DeviceStatus.RUNNING, device.getStatus());
    }

    @Test
    void updateDeviceStatus_shouldUpdateToError() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        device.setStatus(Device.DeviceStatus.RUNNING);
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));
        when(deviceRepository.update(any(Device.class))).thenReturn(device);

        service.updateDeviceStatus("DEV-001", Device.DeviceStatus.ERROR);

        assertEquals(Device.DeviceStatus.ERROR, device.getStatus());
    }

    @Test
    void deviceHeartbeat_shouldUpdateLastHeartbeat() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));
        when(deviceRepository.update(any(Device.class))).thenReturn(device);

        service.deviceHeartbeat("DEV-001");

        verify(deviceRepository).update(device);
    }

    @Test
    void deviceHeartbeat_shouldThrowWhenDeviceNotFound() {
        when(deviceRepository.findByDeviceCode("DEV-999")).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () ->
            service.deviceHeartbeat("DEV-999")
        );
    }

    @Test
    void updateDevicePosition_shouldUpdatePosition() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));
        when(deviceRepository.update(any(Device.class))).thenReturn(device);

        service.updateDevicePosition("DEV-001", 10.0, 20.0, 5.0, 90.0);

        verify(deviceRepository).update(device);
    }

    @Test
    void getWorkshopDevices_shouldReturnWorkshopDevices() {
        Device device1 = new Device();
        device1.setDeviceCode("DEV-001");
        Device device2 = new Device();
        device2.setDeviceCode("DEV-002");
        
        when(deviceRepository.findByWorkshopId("WS001")).thenReturn(List.of(device1, device2));

        List<Device> result = service.getWorkshopDevices("WS001");

        assertEquals(2, result.size());
    }

    @Test
    void getOnlineDevices_shouldReturnOnlineDevices() {
        Device device = new Device();
        device.setStatus(Device.DeviceStatus.ONLINE);
        
        when(deviceRepository.findByStatus(Device.DeviceStatus.ONLINE)).thenReturn(List.of(device));

        List<Device> result = service.getOnlineDevices();

        assertEquals(1, result.size());
        assertEquals(Device.DeviceStatus.ONLINE, result.get(0).getStatus());
    }

    @Test
    void createWorkshop_shouldCreateSuccessfully() {
        Workshop savedWorkshop = new Workshop();
        savedWorkshop.setId(1L);
        savedWorkshop.setWorkshopCode("WS001");
        
        when(workshopRepository.save(any(Workshop.class))).thenReturn(savedWorkshop);

        Workshop result = service.createWorkshop("WS001", "Workshop 1", "Test workshop");

        assertNotNull(result);
        assertEquals("WS001", result.getWorkshopCode());
        verify(workshopRepository).save(any(Workshop.class));
    }

    @Test
    void updateWorkshopStatus_shouldUpdateToOperating() {
        Workshop workshop = new Workshop();
        workshop.setWorkshopCode("WS001");
        workshop.setStatus(Workshop.WorkshopStatus.CONSTRUCTION);
        
        when(workshopRepository.findByWorkshopCode("WS001")).thenReturn(Optional.of(workshop));
        when(workshopRepository.update(any(Workshop.class))).thenReturn(workshop);

        service.updateWorkshopStatus("WS001", Workshop.WorkshopStatus.OPERATING);

        assertEquals(Workshop.WorkshopStatus.OPERATING, workshop.getStatus());
    }

    @Test
    void getAllWorkshops_shouldReturnAllWorkshops() {
        Workshop workshop1 = new Workshop();
        workshop1.setId(1L);
        Workshop workshop2 = new Workshop();
        workshop2.setId(2L);
        
        when(workshopRepository.findAll()).thenReturn(List.of(workshop1, workshop2));

        List<Workshop> result = service.getAllWorkshops();

        assertEquals(2, result.size());
    }

    @Test
    void createProductionLine_shouldCreateSuccessfully() {
        ProductionLine savedLine = new ProductionLine();
        savedLine.setId(1L);
        savedLine.setLineCode("PL001");
        
        when(productionLineRepository.save(any(ProductionLine.class))).thenReturn(savedLine);

        ProductionLine result = service.createProductionLine(
            "PL001", "Production Line 1", "WS001", 100
        );

        assertNotNull(result);
        assertEquals("PL001", result.getLineCode());
        verify(productionLineRepository).save(any(ProductionLine.class));
    }

    @Test
    void updateProductionLineStatus_shouldUpdateToRunning() {
        ProductionLine line = new ProductionLine();
        line.setLineCode("PL001");
        line.setStatus(ProductionLine.ProductionLineStatus.IDLE);
        
        when(productionLineRepository.findByLineCode("PL001")).thenReturn(Optional.of(line));
        when(productionLineRepository.update(any(ProductionLine.class))).thenReturn(line);

        service.updateProductionLineStatus("PL001", ProductionLine.ProductionLineStatus.RUNNING);

        assertEquals(ProductionLine.ProductionLineStatus.RUNNING, line.getStatus());
    }

    @Test
    void updateProductionLineOutput_shouldUpdateOutput() {
        ProductionLine line = new ProductionLine();
        line.setLineCode("PL001");
        line.setCurrentOutput(0);
        
        when(productionLineRepository.findByLineCode("PL001")).thenReturn(Optional.of(line));
        when(productionLineRepository.update(any(ProductionLine.class))).thenReturn(line);

        service.updateProductionLineOutput("PL001", 50);

        assertEquals(50, line.getCurrentOutput());
    }

    @Test
    void getWorkshopProductionLines_shouldReturnLines() {
        ProductionLine line = new ProductionLine();
        line.setLineCode("PL001");
        
        when(productionLineRepository.findByWorkshopId("WS001")).thenReturn(List.of(line));

        List<ProductionLine> result = service.getWorkshopProductionLines("WS001");

        assertEquals(1, result.size());
    }

    @Test
    void createAlert_shouldCreateSuccessfully() {
        Alert savedAlert = new Alert();
        savedAlert.setId(1L);
        savedAlert.setAlertCode("ALT-001");
        
        when(alertRepository.save(any(Alert.class))).thenReturn(savedAlert);

        Alert result = service.createAlert(
            "DEV-001", "WS001", Alert.AlertLevel.INFO,
            Alert.AlertType.DEVICE_OFFLINE, "Test Alert", "Test description"
        );

        assertNotNull(result);
        verify(alertRepository).save(any(Alert.class));
    }

    @Test
    void acknowledgeAlert_shouldUpdateStatus() {
        Alert alert = new Alert();
        alert.setId(1L);
        alert.setStatus(Alert.AlertStatus.TRIGGERED);
        
        when(alertRepository.findById(1L)).thenReturn(Optional.of(alert));
        when(alertRepository.update(any(Alert.class))).thenReturn(alert);

        service.acknowledgeAlert(1L, "admin");

        assertEquals(Alert.AlertStatus.ACKNOWLEDGED, alert.getStatus());
        assertNotNull(alert.getAcknowledgedAt());
    }

    @Test
    void resolveAlert_shouldUpdateStatusAndSolution() {
        Alert alert = new Alert();
        alert.setId(1L);
        alert.setStatus(Alert.AlertStatus.ACKNOWLEDGED);
        
        when(alertRepository.findById(1L)).thenReturn(Optional.of(alert));
        when(alertRepository.update(any(Alert.class))).thenReturn(alert);

        service.resolveAlert(1L, "Fixed the issue", "admin");

        assertEquals(Alert.AlertStatus.RESOLVED, alert.getStatus());
        assertEquals("Fixed the issue", alert.getSolution());
    }

    @Test
    void getActiveAlerts_shouldReturnTriggeredAlerts() {
        Alert alert = new Alert();
        alert.setStatus(Alert.AlertStatus.TRIGGERED);
        
        when(alertRepository.findByStatus(Alert.AlertStatus.TRIGGERED)).thenReturn(List.of(alert));

        List<Alert> result = service.getActiveAlerts();

        assertEquals(1, result.size());
    }

    @Test
    void getDeviceAlerts_shouldReturnDeviceAlerts() {
        Alert alert = new Alert();
        alert.setDeviceCode("DEV-001");
        
        when(alertRepository.findByDeviceCode("DEV-001")).thenReturn(List.of(alert));

        List<Alert> result = service.getDeviceAlerts("DEV-001");

        assertEquals(1, result.size());
    }

    @Test
    void getOnlineDeviceCount_shouldReturnCount() {
        Device device = new Device();
        device.setStatus(Device.DeviceStatus.ONLINE);
        
        when(deviceRepository.findByStatus(Device.DeviceStatus.ONLINE)).thenReturn(List.of(device));

        Long count = service.getOnlineDeviceCount();

        assertEquals(1L, count);
    }

    @Test
    void getActiveAlertCount_shouldReturnCount() {
        Alert alert = new Alert();
        alert.setStatus(Alert.AlertStatus.TRIGGERED);
        
        when(alertRepository.findByStatus(Alert.AlertStatus.TRIGGERED)).thenReturn(List.of(alert));

        Long count = service.getActiveAlertCount();

        assertEquals(1L, count);
    }

    @Test
    void getAverageEfficiency_shouldCalculateAverage() {
        ProductionLine line1 = new ProductionLine();
        line1.setEfficiency(80.0);
        ProductionLine line2 = new ProductionLine();
        line2.setEfficiency(90.0);
        
        when(productionLineRepository.findAll()).thenReturn(List.of(line1, line2));

        Double avg = service.getAverageEfficiency();

        assertEquals(85.0, avg);
    }

    @Test
    void getAverageEfficiency_shouldReturnZeroWhenNoLines() {
        when(productionLineRepository.findAll()).thenReturn(List.of());

        Double avg = service.getAverageEfficiency();

        assertEquals(0.0, avg);
    }
}