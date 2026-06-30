package com.metawebthree.digitaltwin.application.query;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import com.metawebthree.digitaltwin.domain.service.DigitalTwinDomainService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinQueryServiceTest {

    @Mock
    private DeviceRepository deviceRepository;
    @Mock
    private WorkshopRepository workshopRepository;
    @Mock
    private ProductionLineRepository productionLineRepository;
    @Mock
    private AlertRepository alertRepository;
    @Mock
    private DigitalTwinDomainService domainService;

    private DigitalTwinQueryService service;

    @BeforeEach
    void setUp() {
        service = new DigitalTwinQueryService(
            deviceRepository,
            workshopRepository,
            productionLineRepository,
            alertRepository,
            domainService
        );
    }

    @Test
    void getDeviceById_shouldReturnDevice() {
        Device device = new Device();
        device.setId(1L);
        device.setDeviceCode("DEV-001");
        
        when(deviceRepository.findById(1L)).thenReturn(Optional.of(device));

        Optional<Device> result = service.getDeviceById(1L);

        assertTrue(result.isPresent());
        assertEquals("DEV-001", result.get().getDeviceCode());
    }

    @Test
    void getDeviceById_shouldReturnEmptyWhenNotFound() {
        when(deviceRepository.findById(999L)).thenReturn(Optional.empty());

        Optional<Device> result = service.getDeviceById(999L);

        assertFalse(result.isPresent());
    }

    @Test
    void getDeviceByCode_shouldReturnDevice() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        
        when(deviceRepository.findByDeviceCode("DEV-001")).thenReturn(Optional.of(device));

        Optional<Device> result = service.getDeviceByCode("DEV-001");

        assertTrue(result.isPresent());
    }

    @Test
    void getAllDevices_shouldReturnAllDevices() {
        Device device1 = new Device();
        device1.setDeviceCode("DEV-001");
        Device device2 = new Device();
        device2.setDeviceCode("DEV-002");
        
        when(deviceRepository.findAll()).thenReturn(List.of(device1, device2));

        List<Device> result = service.getAllDevices();

        assertEquals(2, result.size());
    }

    @Test
    void getDevicesPaginated_shouldReturnPagedResult() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        
        IPage<Device> page = new Page<>(1, 10);
        page.setRecords(List.of(device));
        page.setTotal(1);
        
        when(deviceRepository.findPaginated(1, 10)).thenReturn(page);

        Map<String, Object> result = service.getDevicesPaginated(1, 10);

        assertNotNull(result);
        assertEquals(1, ((List<?>) result.get("data")).size());
        assertEquals(1, result.get("page"));
        assertEquals(10, result.get("size"));
        assertEquals(1L, result.get("total"));
    }

    @Test
    void getWorkshopDevices_shouldReturnWorkshopDevices() {
        Device device = new Device();
        device.setDeviceCode("DEV-001");
        
        when(deviceRepository.findByWorkshopId("WS001")).thenReturn(List.of(device));

        List<Device> result = service.getWorkshopDevices("WS001");

        assertEquals(1, result.size());
    }

    @Test
    void getOnlineDevices_shouldCallDomainService() {
        Device device = new Device();
        device.setStatus(Device.DeviceStatus.ONLINE);
        
        when(domainService.getOnlineDevices()).thenReturn(List.of(device));

        List<Device> result = service.getOnlineDevices();

        assertEquals(1, result.size());
        verify(domainService).getOnlineDevices();
    }

    @Test
    void getWorkshopById_shouldReturnWorkshop() {
        Workshop workshop = new Workshop();
        workshop.setId(1L);
        workshop.setWorkshopCode("WS001");
        
        when(workshopRepository.findById(1L)).thenReturn(Optional.of(workshop));

        Optional<Workshop> result = service.getWorkshopById(1L);

        assertTrue(result.isPresent());
        assertEquals("WS001", result.get().getWorkshopCode());
    }

    @Test
    void getAllWorkshops_shouldReturnAllWorkshops() {
        Workshop workshop = new Workshop();
        workshop.setWorkshopCode("WS001");
        
        when(workshopRepository.findAll()).thenReturn(List.of(workshop));

        List<Workshop> result = service.getAllWorkshops();

        assertEquals(1, result.size());
    }

    @Test
    void getWorkshopsPaginated_shouldReturnPagedResult() {
        Workshop workshop = new Workshop();
        workshop.setWorkshopCode("WS001");
        
        IPage<Workshop> page = new Page<>(1, 10);
        page.setRecords(List.of(workshop));
        page.setTotal(1);
        
        when(workshopRepository.findPaginated(1, 10)).thenReturn(page);

        Map<String, Object> result = service.getWorkshopsPaginated(1, 10);

        assertNotNull(result);
        assertEquals(1, ((List<?>) result.get("data")).size());
    }

    @Test
    void getProductionLineById_shouldReturnLine() {
        ProductionLine line = new ProductionLine();
        line.setId(1L);
        line.setLineCode("PL001");
        
        when(productionLineRepository.findById(1L)).thenReturn(Optional.of(line));

        Optional<ProductionLine> result = service.getProductionLineById(1L);

        assertTrue(result.isPresent());
        assertEquals("PL001", result.get().getLineCode());
    }

    @Test
    void getAllProductionLines_shouldReturnAllLines() {
        ProductionLine line = new ProductionLine();
        line.setLineCode("PL001");
        
        when(productionLineRepository.findAll()).thenReturn(List.of(line));

        List<ProductionLine> result = service.getAllProductionLines();

        assertEquals(1, result.size());
    }

    @Test
    void getProductionLinesPaginated_shouldReturnPagedResult() {
        ProductionLine line = new ProductionLine();
        line.setLineCode("PL001");
        
        IPage<ProductionLine> page = new Page<>(1, 10);
        page.setRecords(List.of(line));
        page.setTotal(1);
        
        when(productionLineRepository.findPaginated(1, 10)).thenReturn(page);

        Map<String, Object> result = service.getProductionLinesPaginated(1, 10);

        assertNotNull(result);
        assertEquals(1, ((List<?>) result.get("data")).size());
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
    void getAlertById_shouldReturnAlert() {
        Alert alert = new Alert();
        alert.setId(1L);
        alert.setAlertCode("ALT-001");
        
        when(alertRepository.findById(1L)).thenReturn(Optional.of(alert));

        Optional<Alert> result = service.getAlertById(1L);

        assertTrue(result.isPresent());
        assertEquals("ALT-001", result.get().getAlertCode());
    }

    @Test
    void getAllAlerts_shouldReturnAllAlerts() {
        Alert alert = new Alert();
        alert.setAlertCode("ALT-001");
        
        when(alertRepository.findAll()).thenReturn(List.of(alert));

        List<Alert> result = service.getAllAlerts();

        assertEquals(1, result.size());
    }

    @Test
    void getActiveAlerts_shouldCallDomainService() {
        Alert alert = new Alert();
        alert.setStatus(Alert.AlertStatus.TRIGGERED);
        
        when(domainService.getActiveAlerts()).thenReturn(List.of(alert));

        List<Alert> result = service.getActiveAlerts();

        assertEquals(1, result.size());
        verify(domainService).getActiveAlerts();
    }

    @Test
    void getDeviceAlerts_shouldCallDomainService() {
        Alert alert = new Alert();
        alert.setDeviceCode("DEV-001");
        
        when(domainService.getDeviceAlerts("DEV-001")).thenReturn(List.of(alert));

        List<Alert> result = service.getDeviceAlerts("DEV-001");

        assertEquals(1, result.size());
        verify(domainService).getDeviceAlerts("DEV-001");
    }

    @Test
    void getWorkshopAlerts_shouldReturnAlerts() {
        Alert alert = new Alert();
        alert.setWorkshopId("WS001");
        
        when(alertRepository.findByWorkshopId("WS001")).thenReturn(List.of(alert));

        List<Alert> result = service.getWorkshopAlerts("WS001");

        assertEquals(1, result.size());
    }

    @Test
    void getOnlineDeviceCount_shouldCallDomainService() {
        when(domainService.getOnlineDeviceCount()).thenReturn(5L);

        Long count = service.getOnlineDeviceCount();

        assertEquals(5L, count);
        verify(domainService).getOnlineDeviceCount();
    }

    @Test
    void getActiveAlertCount_shouldCallDomainService() {
        when(domainService.getActiveAlertCount()).thenReturn(3L);

        Long count = service.getActiveAlertCount();

        assertEquals(3L, count);
        verify(domainService).getActiveAlertCount();
    }

    @Test
    void getAverageEfficiency_shouldCallDomainService() {
        when(domainService.getAverageEfficiency()).thenReturn(85.5);

        Double avg = service.getAverageEfficiency();

        assertEquals(85.5, avg);
        verify(domainService).getAverageEfficiency();
    }
}