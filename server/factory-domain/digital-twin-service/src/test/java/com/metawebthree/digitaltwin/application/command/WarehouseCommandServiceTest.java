package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;
import com.metawebthree.digitaltwin.domain.repository.WarehouseRepository;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class WarehouseCommandServiceTest {

    @Mock
    private WarehouseRepository warehouseRepository;

    @Mock
    private DigitalTwinEventPublisher eventPublisher;

    private WarehouseCommandService service;

    @BeforeEach
    void setUp() {
        service = new WarehouseCommandService(warehouseRepository, eventPublisher);
    }

    private Warehouse createSampleWarehouse(Long id, String warehouseCode) {
        Warehouse warehouse = new Warehouse(warehouseCode, "Test Warehouse");
        warehouse.setId(id);
        warehouse.setDescription("Test Description");
        warehouse.setStatus(WarehouseStatus.PLANNING);
        warehouse.setTotalArea(BigDecimal.valueOf(1000));
        warehouse.setUsedArea(BigDecimal.ZERO);
        return warehouse;
    }

    private WarehouseCommandService.CreateWarehouseRequest createBaseRequest() {
        WarehouseCommandService.CreateWarehouseRequest request = new WarehouseCommandService.CreateWarehouseRequest();
        request.warehouseCode = "WH-001";
        request.warehouseName = "Test Warehouse";
        request.description = "Test Description";
        request.totalArea = BigDecimal.valueOf(1000);
        request.location = "Test Location";
        request.centerX = BigDecimal.valueOf(10.0);
        request.centerY = BigDecimal.valueOf(20.0);
        request.centerZ = BigDecimal.valueOf(30.0);
        request.width = BigDecimal.valueOf(100);
        request.length = BigDecimal.valueOf(200);
        request.height = BigDecimal.valueOf(50);
        return request;
    }

    @Test
    void createWarehouse_shouldCreateSuccessfully() {
        when(warehouseRepository.existsByWarehouseCode("WH-001")).thenReturn(false);
        when(warehouseRepository.save(any(Warehouse.class))).thenAnswer(invocation -> {
            Warehouse w = invocation.getArgument(0);
            w.setId(1L);
            return w;
        });

        Warehouse result = service.createWarehouse(createBaseRequest());

        assertNotNull(result);
        assertEquals("WH-001", result.getWarehouseCode());
        assertEquals("Test Warehouse", result.getWarehouseName());
        assertEquals(WarehouseStatus.PLANNING, result.getStatus());
        verify(warehouseRepository).save(any(Warehouse.class));
    }

    @Test
    void createWarehouse_shouldThrowExceptionWhenCodeExists() {
        when(warehouseRepository.existsByWarehouseCode("WH-001")).thenReturn(true);

        WarehouseCommandService.CreateWarehouseRequest request = createBaseRequest();
        assertThrows(IllegalArgumentException.class, () -> service.createWarehouse(request));
    }

    @Test
    void updateWarehouse_shouldUpdateSuccessfully() {
        Warehouse warehouse = createSampleWarehouse(1L, "WH-001");
        when(warehouseRepository.findById(1L)).thenReturn(Optional.of(warehouse));
        when(warehouseRepository.save(any(Warehouse.class))).thenReturn(warehouse);

        WarehouseCommandService.UpdateWarehouseRequest request = new WarehouseCommandService.UpdateWarehouseRequest();
        request.id = 1L;
        request.warehouseName = "Updated Name";
        request.description = "Updated Description";

        Warehouse result = service.updateWarehouse(request);

        assertNotNull(result);
        assertEquals("Updated Name", result.getWarehouseName());
        verify(warehouseRepository).save(any(Warehouse.class));
    }

    @Test
    void updateWarehouse_shouldThrowExceptionWhenNotFound() {
        when(warehouseRepository.findById(1L)).thenReturn(Optional.empty());

        WarehouseCommandService.UpdateWarehouseRequest request = new WarehouseCommandService.UpdateWarehouseRequest();
        request.id = 1L;

        assertThrows(IllegalArgumentException.class, () -> service.updateWarehouse(request));
    }

    @Test
    void activateWarehouse_shouldActivateAndPublishEvent() {
        Warehouse warehouse = createSampleWarehouse(1L, "WH-001");
        when(warehouseRepository.findById(1L)).thenReturn(Optional.of(warehouse));
        when(warehouseRepository.save(any(Warehouse.class))).thenReturn(warehouse);

        service.activateWarehouse(1L);

        assertEquals(WarehouseStatus.OPERATING, warehouse.getStatus());
        verify(warehouseRepository).save(warehouse);
        verify(eventPublisher).publishWarehouseStatusChanged("WH-001", "OPERATING");
    }

    @Test
    void activateWarehouse_shouldThrowExceptionWhenNotFound() {
        when(warehouseRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.activateWarehouse(1L));
    }

    @Test
    void decommissionWarehouse_shouldDecommissionAndPublishEvent() {
        Warehouse warehouse = createSampleWarehouse(1L, "WH-001");
        when(warehouseRepository.findById(1L)).thenReturn(Optional.of(warehouse));
        when(warehouseRepository.save(any(Warehouse.class))).thenReturn(warehouse);

        service.decommissionWarehouse(1L);

        assertEquals(WarehouseStatus.DECOMMISSIONED, warehouse.getStatus());
        verify(warehouseRepository).save(warehouse);
        verify(eventPublisher).publishWarehouseStatusChanged("WH-001", "DECOMMISSIONED");
    }

    @Test
    void deleteWarehouse_shouldDeleteSuccessfully() {
        Warehouse warehouse = createSampleWarehouse(1L, "WH-001");
        when(warehouseRepository.findById(1L)).thenReturn(Optional.of(warehouse));
        doNothing().when(warehouseRepository).delete(warehouse);

        service.deleteWarehouse(1L);

        verify(warehouseRepository).delete(warehouse);
    }

    @Test
    void deleteWarehouse_shouldThrowExceptionWhenNotFound() {
        when(warehouseRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.deleteWarehouse(1L));
    }
}