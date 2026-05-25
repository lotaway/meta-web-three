package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.*;

class InventoryItemTest {

    private InventoryItem item;

    @BeforeEach
    void setUp() {
        item = new InventoryItem("ITEM-001", "SKU-001", "Test Product");
    }

    @Test
    void constructor_shouldInitializeWithOutOfStockStatus() {
        assertEquals("ITEM-001", item.getItemCode());
        assertEquals("SKU-001", item.getSku());
        assertEquals("Test Product", item.getItemName());
        assertEquals(BigDecimal.ZERO, item.getQuantity());
        assertEquals(InventoryItem.ItemStatus.OUT_OF_STOCK, item.getStatus());
        assertNotNull(item.getCreatedAt());
    }

    @Test
    void addQuantity_shouldIncreaseQuantity() {
        item.addQuantity(new BigDecimal("100"));
        
        assertEquals(new BigDecimal("100"), item.getQuantity());
    }

    @Test
    void addQuantity_shouldUpdateStatusToNormal() {
        item.setMinQuantity(new BigDecimal("50"));
        item.addQuantity(new BigDecimal("100"));
        
        assertEquals(InventoryItem.ItemStatus.NORMAL, item.getStatus());
    }

    @Test
    void removeQuantity_shouldDecreaseQuantity() {
        item.setQuantity(new BigDecimal("100"));
        
        item.removeQuantity(new BigDecimal("30"));
        
        assertEquals(new BigDecimal("70"), item.getQuantity());
    }

    @Test
    void removeQuantity_shouldNotGoBelowZero() {
        item.setQuantity(new BigDecimal("50"));
        
        item.removeQuantity(new BigDecimal("100"));
        
        assertEquals(BigDecimal.ZERO, item.getQuantity());
    }

    @Test
    void updateStatus_shouldSetOutOfStockWhenQuantityIsZero() {
        item.setQuantity(BigDecimal.ZERO);
        
        item.updateStatus();
        
        assertEquals(InventoryItem.ItemStatus.OUT_OF_STOCK, item.getStatus());
    }

    @Test
    void updateStatus_shouldSetCriticalWhenBelowHalfMinQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setQuantity(new BigDecimal("40"));
        
        item.updateStatus();
        
        assertEquals(InventoryItem.ItemStatus.CRITICAL, item.getStatus());
    }

    @Test
    void updateStatus_shouldSetLowWhenBelowMinQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setQuantity(new BigDecimal("60"));
        
        item.updateStatus();
        
        assertEquals(InventoryItem.ItemStatus.LOW, item.getStatus());
    }

    @Test
    void updateStatus_shouldSetNormalWhenAboveMinQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setQuantity(new BigDecimal("150"));
        
        item.updateStatus();
        
        assertEquals(InventoryItem.ItemStatus.NORMAL, item.getStatus());
    }

    @Test
    void updateStatus_shouldSetExpiredWhenPastExpiryDate() {
        item.setExpiryDate(LocalDate.now().minusDays(1));
        item.setQuantity(new BigDecimal("100"));
        
        item.updateStatus();
        
        assertEquals(InventoryItem.ItemStatus.EXPIRED, item.getStatus());
    }

    @Test
    void needsRestock_shouldReturnTrueWhenBelowMinQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setQuantity(new BigDecimal("80"));
        
        assertTrue(item.needsRestock());
    }

    @Test
    void needsRestock_shouldReturnFalseWhenAboveMinQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setQuantity(new BigDecimal("150"));
        
        assertFalse(item.needsRestock());
    }

    @Test
    void needsRestock_shouldReturnFalseWhenMinQuantityIsNull() {
        item.setMinQuantity(null);
        item.setQuantity(new BigDecimal("80"));
        
        assertFalse(item.needsRestock());
    }

    @Test
    void isExpiringSoon_shouldReturnTrueWhenWithinThreshold() {
        item.setExpiryDate(LocalDate.now().plusDays(5));
        
        assertTrue(item.isExpiringSoon(7));
    }

    @Test
    void isExpiringSoon_shouldReturnFalseWhenBeyondThreshold() {
        item.setExpiryDate(LocalDate.now().plusDays(10));
        
        assertFalse(item.isExpiringSoon(7));
    }

    @Test
    void isExpiringSoon_shouldReturnFalseWhenNoExpiryDate() {
        item.setExpiryDate(null);
        
        assertFalse(item.isExpiringSoon(7));
    }

    @Test
    void calculateReorderQuantity_shouldCalculateCorrectAmount() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setMaxQuantity(new BigDecimal("500"));
        item.setQuantity(new BigDecimal("80"));
        
        BigDecimal reorderQty = item.calculateReorderQuantity();
        
        // Should order up to max, so max - current = 500 - 80 = 420
        assertEquals(new BigDecimal("420"), reorderQty);
    }

    @Test
    void calculateReorderQuantity_shouldReturnZeroWhenAtMax() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setMaxQuantity(new BigDecimal("500"));
        item.setQuantity(new BigDecimal("500"));
        
        assertEquals(BigDecimal.ZERO, item.calculateReorderQuantity());
    }

    @Test
    void calculateReorderQuantity_shouldReturnZeroWhenNoMaxQuantity() {
        item.setMinQuantity(new BigDecimal("100"));
        item.setMaxQuantity(null);
        item.setQuantity(new BigDecimal("80"));
        
        assertEquals(BigDecimal.ZERO, item.calculateReorderQuantity());
    }
}