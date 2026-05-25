package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;

class InventoryAlertTest {

    private InventoryAlert alert;

    @BeforeEach
    void setUp() {
        alert = new InventoryAlert(
            "ALERT-001", 
            "ITEM-001", 
            InventoryAlert.AlertType.LOW_STOCK, 
            InventoryAlert.AlertLevel.WARNING, 
            "Low Stock Alert"
        );
    }

    @Test
    void constructor_shouldInitializeWithTriggeredStatus() {
        assertEquals("ALERT-001", alert.getAlertCode());
        assertEquals("ITEM-001", alert.getItemCode());
        assertEquals(InventoryAlert.AlertType.LOW_STOCK, alert.getAlertType());
        assertEquals(InventoryAlert.AlertLevel.WARNING, alert.getAlertLevel());
        assertEquals("Low Stock Alert", alert.getTitle());
        assertEquals(InventoryAlert.AlertStatus.TRIGGERED, alert.getStatus());
        assertNotNull(alert.getOccurredAt());
        assertNotNull(alert.getCreatedAt());
    }

    @Test
    void acknowledge_shouldSetStatusAndUser() {
        alert.acknowledge("admin");
        
        assertEquals(InventoryAlert.AlertStatus.ACKNOWLEDGED, alert.getStatus());
        assertEquals("admin", alert.getAcknowledgedBy());
        assertNotNull(alert.getAcknowledgedAt());
    }

    @Test
    void startResolution_shouldSetStatusToInProgress() {
        alert.acknowledge("admin");
        alert.startResolution();
        
        assertEquals(InventoryAlert.AlertStatus.IN_PROGRESS, alert.getStatus());
    }

    @Test
    void resolve_shouldSetStatusToResolved() {
        alert.acknowledge("admin");
        alert.startResolution();
        alert.resolve("admin", "Added more stock");
        
        assertEquals(InventoryAlert.AlertStatus.RESOLVED, alert.getStatus());
        assertEquals("Added more stock", alert.getSolution());
        assertEquals("admin", alert.getResolvedBy());
        assertNotNull(alert.getResolvedAt());
    }

    @Test
    void close_shouldSetStatusToClosed() {
        alert.close();
        
        assertEquals(InventoryAlert.AlertStatus.CLOSED, alert.getStatus());
    }

    @Test
    void isActive_shouldReturnTrueWhenTriggered() {
        assertTrue(alert.isActive());
    }

    @Test
    void isActive_shouldReturnTrueWhenAcknowledged() {
        alert.acknowledge("admin");
        
        assertTrue(alert.isActive());
    }

    @Test
    void isActive_shouldReturnTrueWhenInProgress() {
        alert.acknowledge("admin");
        alert.startResolution();
        
        assertTrue(alert.isActive());
    }

    @Test
    void isActive_shouldReturnFalseWhenResolved() {
        alert.acknowledge("admin");
        alert.resolve("admin", "Fixed");
        
        assertFalse(alert.isActive());
    }

    @Test
    void isActive_shouldReturnFalseWhenClosed() {
        alert.close();
        
        assertFalse(alert.isActive());
    }

    @Test
    void isAcknowledged_shouldReturnCorrectStatus() {
        assertFalse(alert.isAcknowledged());
        
        alert.acknowledge("admin");
        
        assertTrue(alert.isAcknowledged());
    }

    @Test
    void getResolutionTime_shouldCalculateCorrectDuration() {
        LocalDateTime startTime = LocalDateTime.now().minusHours(2);
        alert.setAcknowledgedAt(startTime);
        alert.resolve("admin", "Fixed");
        
        Long minutes = alert.getResolutionTimeMinutes();
        
        assertNotNull(minutes);
        assertTrue(minutes >= 118 && minutes <= 122); // Allow 2 min variance
    }

    @Test
    void getResolutionTime_shouldReturnNullWhenNotResolved() {
        alert.acknowledge("admin");
        
        assertNull(alert.getResolutionTimeMinutes());
    }

    @Test
    void shouldAutoEscalate_shouldReturnTrueForCriticalLevel() {
        alert.setLevel(InventoryAlert.AlertLevel.CRITICAL);
        
        assertTrue(alert.shouldAutoEscalate());
    }

    @Test
    void shouldAutoEscalate_shouldReturnTrueForErrorNotAcknowledged() {
        alert.setLevel(InventoryAlert.AlertLevel.ERROR);
        alert.setAcknowledgedAt(null);
        
        assertTrue(alert.shouldAutoEscalate());
    }

    @Test
    void shouldAutoEscalate_shouldReturnFalseForInfoAcknowledged() {
        alert.setLevel(InventoryAlert.AlertLevel.INFO);
        alert.acknowledge("admin");
        
        assertFalse(alert.shouldAutoEscalate());
    }

    @Test
    void isOverdue_shouldReturnTrueWhenBeyondThreshold() {
        alert.setAcknowledgedAt(LocalDateTime.now().minusHours(25));
        
        assertTrue(alert.isOverdue(24));
    }

    @Test
    void isOverdue_shouldReturnFalseWhenWithinThreshold() {
        alert.setAcknowledgedAt(LocalDateTime.now().minusHours(12));
        
        assertFalse(alert.isOverdue(24));
    }

    @Test
    void isOverdue_shouldReturnFalseWhenNotAcknowledged() {
        alert.setAcknowledgedAt(null);
        
        assertFalse(alert.isOverdue(24));
    }

    @Test
    void getSeverityScore_shouldReturnCorrectValue() {
        assertEquals(4, alert.getSeverityScore()); // CRITICAL=4, ERROR=3, WARNING=2, INFO=1
        
        alert.setLevel(InventoryAlert.AlertLevel.CRITICAL);
        assertEquals(4, alert.getSeverityScore());
        
        alert.setLevel(InventoryAlert.AlertLevel.ERROR);
        assertEquals(3, alert.getSeverityScore());
        
        alert.setLevel(InventoryAlert.AlertLevel.WARNING);
        assertEquals(2, alert.getSeverityScore());
        
        alert.setLevel(InventoryAlert.AlertLevel.INFO);
        assertEquals(1, alert.getSeverityScore());
    }
}