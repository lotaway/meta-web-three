package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
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
class ShelfCommandServiceTest {

    @Mock
    private ShelfRepository shelfRepository;

    @Mock
    private WarehouseRepository warehouseRepository;

    @Mock
    private DigitalTwinEventPublisher eventPublisher;

    private ShelfCommandService service;

    @BeforeEach
    void setUp() {
        service = new ShelfCommandService(shelfRepository, warehouseRepository, eventPublisher);
    }

    private Shelf createSampleShelf(Long id, String shelfCode, String warehouseCode) {
        Shelf shelf = new Shelf(shelfCode, warehouseCode, 1, 1);
        shelf.setId(id);
        shelf.setZone("A");
        shelf.setLevelNumber(3);
        shelf.setStatus(ShelfStatus.EMPTY);
        shelf.setMaxWeight(BigDecimal.valueOf(1000));
        shelf.setCurrentWeight(BigDecimal.ZERO);
        return shelf;
    }

    private ShelfCommandService.CreateShelfRequest createBaseRequest() {
        ShelfCommandService.CreateShelfRequest request = new ShelfCommandService.CreateShelfRequest();
        request.shelfCode = "SHELF-001";
        request.warehouseCode = "WH-001";
        request.zone = "A";
        request.rowNumber = 1;
        request.columnNumber = 1;
        request.totalLevels = 5;
        request.maxWeight = BigDecimal.valueOf(1000);
        request.positionX = BigDecimal.valueOf(10.0);
        request.positionY = BigDecimal.valueOf(20.0);
        request.positionZ = BigDecimal.valueOf(30.0);
        request.rotationY = BigDecimal.valueOf(0.0);
        request.length = BigDecimal.valueOf(2.0);
        request.width = BigDecimal.valueOf(1.0);
        request.height = BigDecimal.valueOf(3.0);
        return request;
    }

    @Test
    void createShelf_shouldCreateSuccessfully() {
        when(shelfRepository.existsByShelfCode("SHELF-001")).thenReturn(false);
        when(warehouseRepository.existsByWarehouseCode("WH-001")).thenReturn(true);
        doAnswer(invocation -> {
            Shelf s = invocation.getArgument(0);
            s.setId(1L);
            return null;
        }).when(shelfRepository).insert(any(Shelf.class));

        Shelf result = service.createShelf(createBaseRequest());

        assertNotNull(result);
        assertEquals("SHELF-001", result.getShelfCode());
        assertEquals("WH-001", result.getWarehouseCode());
        assertEquals(ShelfStatus.EMPTY, result.getStatus());
        verify(shelfRepository).insert(any(Shelf.class));
    }

    @Test
    void createShelf_shouldThrowExceptionWhenShelfCodeExists() {
        when(shelfRepository.existsByShelfCode("SHELF-001")).thenReturn(true);

        assertThrows(IllegalArgumentException.class, () -> service.createShelf(createBaseRequest()));
    }

    @Test
    void createShelf_shouldThrowExceptionWhenWarehouseNotFound() {
        when(shelfRepository.existsByShelfCode("SHELF-001")).thenReturn(false);
        when(warehouseRepository.existsByWarehouseCode("WH-001")).thenReturn(false);

        assertThrows(IllegalArgumentException.class, () -> service.createShelf(createBaseRequest()));
    }

    @Test
    void updateShelf_shouldUpdateSuccessfully() {
        Shelf shelf = createSampleShelf(1L, "SHELF-001", "WH-001");
        when(shelfRepository.findById(1L)).thenReturn(Optional.of(shelf));
        doNothing().when(shelfRepository).update(any(Shelf.class));

        ShelfCommandService.UpdateShelfRequest request = new ShelfCommandService.UpdateShelfRequest();
        request.id = 1L;
        request.zone = "B";
        request.maxWeight = BigDecimal.valueOf(1500);

        Shelf result = service.updateShelf(request);

        assertNotNull(result);
        assertEquals("B", result.getZone());
        verify(shelfRepository).update(any(Shelf.class));
    }

    @Test
    void updateShelf_shouldThrowExceptionWhenNotFound() {
        when(shelfRepository.findById(1L)).thenReturn(Optional.empty());

        ShelfCommandService.UpdateShelfRequest request = new ShelfCommandService.UpdateShelfRequest();
        request.id = 1L;

        assertThrows(IllegalArgumentException.class, () -> service.updateShelf(request));
    }

    @Test
    void occupyShelf_shouldOccupyAndPublishEvent() {
        Shelf shelf = createSampleShelf(1L, "SHELF-001", "WH-001");
        when(shelfRepository.findById(1L)).thenReturn(Optional.of(shelf));
        doNothing().when(shelfRepository).update(any(Shelf.class));

        service.occupyShelf(1L);

        assertEquals(ShelfStatus.OCCUPIED, shelf.getStatus());
        verify(shelfRepository).update(shelf);
        verify(eventPublisher).publishShelfStatusChanged("WH-001", "SHELF-001", "OCCUPIED");
    }

    @Test
    void occupyShelf_shouldThrowExceptionWhenNotFound() {
        when(shelfRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.occupyShelf(1L));
    }

    @Test
    void clearShelf_shouldClearAndPublishEvent() {
        Shelf shelf = createSampleShelf(1L, "SHELF-001", "WH-001");
        shelf.setStatus(ShelfStatus.OCCUPIED);
        when(shelfRepository.findById(1L)).thenReturn(Optional.of(shelf));
        doNothing().when(shelfRepository).update(any(Shelf.class));

        service.clearShelf(1L);

        assertEquals(ShelfStatus.EMPTY, shelf.getStatus());
        verify(shelfRepository).update(shelf);
        verify(eventPublisher).publishShelfStatusChanged("WH-001", "SHELF-001", "EMPTY");
    }

    @Test
    void clearShelf_shouldThrowExceptionWhenNotFound() {
        when(shelfRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.clearShelf(1L));
    }

    @Test
    void deleteShelf_shouldDeleteSuccessfully() {
        Shelf shelf = createSampleShelf(1L, "SHELF-001", "WH-001");
        when(shelfRepository.findById(1L)).thenReturn(Optional.of(shelf));
        doNothing().when(shelfRepository).delete(shelf);

        service.deleteShelf(1L);

        verify(shelfRepository).delete(shelf);
    }

    @Test
    void deleteShelf_shouldThrowExceptionWhenNotFound() {
        when(shelfRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.deleteShelf(1L));
    }
}