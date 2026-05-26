package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertType;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.repository.InventoryAlertRepository;
import com.metawebthree.digitaltwin.domain.repository.InventoryItemRepository;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class InventoryCommandServiceTest {

    @Mock
    private InventoryItemRepository inventoryItemRepository;

    @Mock
    private InventoryAlertRepository inventoryAlertRepository;

    @Mock
    private ShelfRepository shelfRepository;

    @Mock
    private DigitalTwinEventPublisher eventPublisher;

    private InventoryCommandService service;

    @BeforeEach
    void setUp() {
        service = new InventoryCommandService(
            inventoryItemRepository,
            inventoryAlertRepository,
            shelfRepository,
            eventPublisher
        );
    }

    private InventoryItem createSampleItem(Long id, String itemCode) {
        InventoryItem item = new InventoryItem(itemCode, "SKU-001", "Test Item");
        item.setId(id);
        item.setCategory("Electronics");
        item.setUnit("PCS");
        item.setQuantity(BigDecimal.valueOf(100));
        item.setMinQuantity(BigDecimal.valueOf(20));
        item.setMaxQuantity(BigDecimal.valueOf(500));
        item.setShelfCode("SHELF-001");
        return item;
    }

    private InventoryCommandService.CreateItemRequest createBaseRequest() {
        InventoryCommandService.CreateItemRequest request = new InventoryCommandService.CreateItemRequest();
        request.itemCode = "ITEM-001";
        request.sku = "SKU-001";
        request.itemName = "Test Item";
        request.category = "Electronics";
        request.unit = "PCS";
        request.quantity = BigDecimal.valueOf(100);
        request.minQuantity = BigDecimal.valueOf(20);
        request.maxQuantity = BigDecimal.valueOf(500);
        request.shelfCode = "SHELF-001";
        request.batchNumber = "BATCH-001";
        request.productionDate = LocalDate.now().minusMonths(1);
        request.expiryDate = LocalDate.now().plusMonths(11);
        request.unitPrice = BigDecimal.valueOf(99.99);
        return request;
    }

    @Test
    void createItem_shouldCreateSuccessfully() {
        when(inventoryItemRepository.existsByItemCode("ITEM-001")).thenReturn(false);
        when(shelfRepository.existsByShelfCode("SHELF-001")).thenReturn(true);
        doAnswer(invocation -> {
            InventoryItem item = invocation.getArgument(0);
            item.setId(1L);
            return null;
        }).when(inventoryItemRepository).insert(any(InventoryItem.class));

        InventoryItem result = service.createItem(createBaseRequest());

        assertNotNull(result);
        assertEquals("ITEM-001", result.getItemCode());
        assertEquals("SKU-001", result.getSku());
        verify(inventoryItemRepository).insert(any(InventoryItem.class));
    }

    @Test
    void createItem_shouldThrowExceptionWhenItemCodeExists() {
        when(inventoryItemRepository.existsByItemCode("ITEM-001")).thenReturn(true);

        assertThrows(IllegalArgumentException.class, () -> service.createItem(createBaseRequest()));
    }

    @Test
    void createItem_shouldThrowExceptionWhenShelfNotFound() {
        when(inventoryItemRepository.existsByItemCode("ITEM-001")).thenReturn(false);
        when(shelfRepository.existsByShelfCode("SHELF-001")).thenReturn(false);

        assertThrows(IllegalArgumentException.class, () -> service.createItem(createBaseRequest()));
    }

    @Test
    void updateItem_shouldUpdateSuccessfully() {
        InventoryItem item = createSampleItem(1L, "ITEM-001");
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.of(item));
        doNothing().when(inventoryItemRepository).update(any(InventoryItem.class));

        InventoryCommandService.UpdateItemRequest request = new InventoryCommandService.UpdateItemRequest();
        request.id = 1L;
        request.itemName = "Updated Item Name";
        request.category = "Updated Category";

        InventoryItem result = service.updateItem(request);

        assertNotNull(result);
        verify(inventoryItemRepository).update(any(InventoryItem.class));
    }

    @Test
    void updateItem_shouldThrowExceptionWhenNotFound() {
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.empty());

        InventoryCommandService.UpdateItemRequest request = new InventoryCommandService.UpdateItemRequest();
        request.id = 1L;

        assertThrows(IllegalArgumentException.class, () -> service.updateItem(request));
    }

    @Test
    void addStock_shouldAddQuantityAndPublishEvent() {
        InventoryItem item = createSampleItem(1L, "ITEM-001");
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.of(item));
        doNothing().when(inventoryItemRepository).update(any(InventoryItem.class));
        when(shelfRepository.findByShelfCode("SHELF-001")).thenReturn(Optional.of(new Shelf("SHELF-001", "WH-001", 1, 1)));

        InventoryItem result = service.addStock(1L, BigDecimal.valueOf(50));

        assertNotNull(result);
        verify(inventoryItemRepository).update(item);
        verify(eventPublisher).publishInventoryLevelChanged(eq("WH-001"), eq("SKU-001"), anyInt(), anyString());
    }

    @Test
    void addStock_shouldThrowExceptionWhenNotFound() {
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.addStock(1L, BigDecimal.valueOf(50)));
    }

    @Test
    void removeStock_shouldRemoveQuantityAndPublishEvent() {
        InventoryItem item = createSampleItem(1L, "ITEM-001");
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.of(item));
        doNothing().when(inventoryItemRepository).update(any(InventoryItem.class));
        when(shelfRepository.findByShelfCode("SHELF-001")).thenReturn(Optional.of(new Shelf("SHELF-001", "WH-001", 1, 1)));

        InventoryItem result = service.removeStock(1L, BigDecimal.valueOf(30));

        assertNotNull(result);
        verify(inventoryItemRepository).update(item);
        verify(eventPublisher).publishInventoryLevelChanged(eq("WH-001"), eq("SKU-001"), anyInt(), anyString());
    }

    @Test
    void removeStock_shouldThrowExceptionWhenNotFound() {
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.removeStock(1L, BigDecimal.valueOf(30)));
    }

    @Test
    void removeStock_shouldCreateLowStockAlertWhenBelowMinimum() {
        InventoryItem item = createSampleItem(1L, "ITEM-001");
        item.setQuantity(BigDecimal.valueOf(15));
        item.setMinQuantity(BigDecimal.valueOf(20));
        
        when(inventoryItemRepository.findById(1L)).thenReturn(Optional.of(item));
        doNothing().when(inventoryItemRepository).update(any(InventoryItem.class));
        when(shelfRepository.findByShelfCode("SHELF-001")).thenReturn(Optional.of(new Shelf("SHELF-001", "WH-001", 1, 1)));
        doAnswer(invocation -> {
            InventoryAlert alert = invocation.getArgument(0);
            alert.setId(1L);
            return null;
        }).when(inventoryAlertRepository).insert(any(InventoryAlert.class));

        InventoryItem result = service.removeStock(1L, BigDecimal.valueOf(10));

        assertNotNull(result);
        verify(inventoryAlertRepository).insert(any(InventoryAlert.class));
        verify(eventPublisher).publishInventoryAlertCreated(eq("WH-001"), anyString(), eq("WARNING"), anyString());
    }

    @Test
    void acknowledgeAlert_shouldAcknowledgeSuccessfully() {
        InventoryAlert alert = new InventoryAlert("ALERT-001", "ITEM-001", AlertType.LOW_STOCK, AlertLevel.WARNING, "Test alert");
        alert.setId(1L);
        
        when(inventoryAlertRepository.findById(1L)).thenReturn(Optional.of(alert));
        doNothing().when(inventoryAlertRepository).update(any(InventoryAlert.class));

        InventoryAlert result = service.acknowledgeAlert(1L, "admin");

        assertNotNull(result);
        verify(inventoryAlertRepository).update(alert);
    }

    @Test
    void acknowledgeAlert_shouldThrowExceptionWhenNotFound() {
        when(inventoryAlertRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.acknowledgeAlert(1L, "admin"));
    }

    @Test
    void resolveAlert_shouldResolveSuccessfully() {
        InventoryAlert alert = new InventoryAlert("ALERT-001", "ITEM-001", AlertType.LOW_STOCK, AlertLevel.WARNING, "Test alert");
        alert.setId(1L);
        
        when(inventoryAlertRepository.findById(1L)).thenReturn(Optional.of(alert));
        doNothing().when(inventoryAlertRepository).update(any(InventoryAlert.class));

        InventoryAlert result = service.resolveAlert(1L, "admin", "Resolved by adding stock");

        assertNotNull(result);
        verify(inventoryAlertRepository).update(alert);
    }

    @Test
    void resolveAlert_shouldThrowExceptionWhenNotFound() {
        when(inventoryAlertRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.resolveAlert(1L, "admin", "Solution"));
    }
}