package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinCommandServiceTest {

    @Mock
    private DigitalTwinDomainService domainService;
    @Mock
    private DigitalTwinEventPublisher eventPublisher;
    @Mock
    private DigitalTwinWebSocketHandler webSocketHandler;

    private DigitalTwinCommandService service;

    @BeforeEach
    void setUp() {
        service = new DigitalTwinCommandService(domainService, eventPublisher, webSocketHandler);
    }

    @Test
    void registerDevice_shouldCreateDeviceAndPublishEvent() {
        Device device = new Device();
        device.setId(1L);
        device.setDeviceCode("DEV-001");
        
        when(domainService.registerDevice(anyString(), anyString(), anyString(), anyString(), anyString()))
            .thenReturn(device);

        Long result = service.registerDevice("DEV-001", "Test Device", "SENSOR", "WS001", "PL001");

        assertEquals(Long.valueOf(1L), result);
        verify(domainService).registerDevice("DEV-001", "Test Device", "SENSOR", "WS001", "PL001");
        verify(eventPublisher).publishDeviceRegistered("DEV-001");
    }

    @Test
    void updateDeviceStatus_shouldUpdateAndBroadcast() {
        doNothing().when(domainService).updateDeviceStatus(anyString(), any());

        service.updateDeviceStatus("DEV-001", Device.DeviceStatus.ONLINE);

        verify(domainService).updateDeviceStatus("DEV-001", Device.DeviceStatus.ONLINE);
        verify(eventPublisher).publishDeviceStatusChanged("DEV-001", "ONLINE");
        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void deviceHeartbeat_shouldCallDomainService() {
        doNothing().when(domainService).deviceHeartbeat(anyString());

        service.deviceHeartbeat("DEV-001");

        verify(domainService).deviceHeartbeat("DEV-001");
    }

    @Test
    void updateDevicePosition_shouldUpdateAndBroadcast() {
        doNothing().when(domainService).updateDevicePosition(anyString(), anyDouble(), anyDouble(), anyDouble(), anyDouble());

        service.updateDevicePosition("DEV-001", 10.0, 20.0, 5.0, 90.0);

        verify(domainService).updateDevicePosition("DEV-001", 10.0, 20.0, 5.0, 90.0);
        verify(eventPublisher).publishDevicePositionUpdated("DEV-001", 10.0, 20.0, 5.0);
        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void createWorkshop_shouldCreateAndPublishEvent() {
        Workshop workshop = new Workshop();
        workshop.setId(1L);
        workshop.setWorkshopCode("WS001");
        
        when(domainService.createWorkshop(anyString(), anyString(), anyString()))
            .thenReturn(workshop);

        Long result = service.createWorkshop("WS001", "Workshop 1", "Test");

        assertEquals(1L, result);
        verify(domainService).createWorkshop("WS001", "Workshop 1", "Test");
        verify(eventPublisher).publishWorkshopCreated("WS001");
    }

    @Test
    void createProductionLine_shouldCreateAndPublishEvent() {
        ProductionLine line = new ProductionLine();
        line.setId(1L);
        line.setLineCode("PL001");
        
        when(domainService.createProductionLine(anyString(), anyString(), anyString(), anyInt()))
            .thenReturn(line);

        Long result = service.createProductionLine("PL001", "Line 1", "WS001", 100);

        assertEquals(1L, result);
        verify(domainService).createProductionLine("PL001", "Line 1", "WS001", 100);
        verify(eventPublisher).publishProductionLineCreated("PL001");
    }

    @Test
    void updateProductionLineOutput_shouldUpdateAndBroadcast() {
        doNothing().when(domainService).updateProductionLineOutput(anyString(), anyInt());

        service.updateProductionLineOutput("PL001", 50);

        verify(domainService).updateProductionLineOutput("PL001", 50);
        verify(eventPublisher).publishProductionOutputUpdated("PL001", 50);
        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void createAlert_shouldCreateAndBroadcast() {
        Alert alert = new Alert();
        alert.setId(1L);
        alert.setAlertCode("ALT-001");
        
        when(domainService.createAlert(anyString(), anyString(), any(), any(), anyString(), anyString()))
            .thenReturn(alert);

        Long result = service.createAlert(
            "DEV-001", "WS001", Alert.AlertLevel.ERROR,
            Alert.AlertType.DEVICE_ERROR, "Error Alert", "Test description"
        );

        assertEquals(1L, result);
        verify(domainService).createAlert(
            "DEV-001", "WS001", Alert.AlertLevel.ERROR,
            Alert.AlertType.DEVICE_ERROR, "Error Alert", "Test description"
        );
        verify(eventPublisher).publishAlertCreated("ALT-001", "ERROR");
        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void acknowledgeAlert_shouldUpdateAndPublishEvent() {
        doNothing().when(domainService).acknowledgeAlert(anyLong(), anyString());

        service.acknowledgeAlert(1L, "admin");

        verify(domainService).acknowledgeAlert(1L, "admin");
        verify(eventPublisher).publishAlertAcknowledged(1L);
    }

    @Test
    void resolveAlert_shouldUpdateAndPublishEvent() {
        doNothing().when(domainService).resolveAlert(anyLong(), anyString(), anyString());

        service.resolveAlert(1L, "Fixed", "admin");

        verify(domainService).resolveAlert(1L, "Fixed", "admin");
        verify(eventPublisher).publishAlertResolved(1L);
    }
}