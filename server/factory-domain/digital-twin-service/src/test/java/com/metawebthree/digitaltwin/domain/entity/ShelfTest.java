package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

class ShelfTest {

    private Shelf shelf;

    @BeforeEach
    void setUp() {
        shelf = new Shelf("SHELF-001", "WH-001", 1, 1);
    }

    @Test
    void constructor_shouldInitializeWithEmptyStatus() {
        assertEquals("SHELF-001", shelf.getShelfCode());
        assertEquals("WH-001", shelf.getWarehouseCode());
        assertEquals(Shelf.ShelfStatus.EMPTY, shelf.getStatus());
        assertEquals(1, shelf.getLevelNumber());
        assertEquals(3, shelf.getTotalLevels());
        assertEquals(BigDecimal.ZERO, shelf.getCurrentWeight());
        assertNotNull(shelf.getCreatedAt());
    }

    @Test
    void occupy_shouldSetStatusToOccupied() {
        shelf.occupy();
        
        assertEquals(Shelf.ShelfStatus.OCCUPIED, shelf.getStatus());
    }

    @Test
    void setFull_shouldSetStatusToFull() {
        shelf.setFull();
        
        assertEquals(Shelf.ShelfStatus.FULL, shelf.getStatus());
    }

    @Test
    void clear_shouldResetStatusAndWeight() {
        shelf.setFull();
        shelf.setMaxWeight(new BigDecimal("1000"));
        shelf.addWeight(new BigDecimal("500"));
        
        shelf.clear();
        
        assertEquals(Shelf.ShelfStatus.EMPTY, shelf.getStatus());
        assertEquals(BigDecimal.ZERO, shelf.getCurrentWeight());
    }

    @Test
    void enterMaintenance_shouldSetStatusToMaintenance() {
        shelf.enterMaintenance();
        
        assertEquals(Shelf.ShelfStatus.MAINTENANCE, shelf.getStatus());
    }

    @Test
    void canAccommodateWeight_shouldReturnTrueWhenNoMaxWeight() {
        shelf.setMaxWeight(null);
        
        assertTrue(shelf.canAccommodateWeight(new BigDecimal("100")));
    }

    @Test
    void canAccommodateWeight_shouldReturnTrueWhenWithinLimit() {
        shelf.setMaxWeight(new BigDecimal("1000"));
        shelf.setCurrentWeight(new BigDecimal("500"));
        
        assertTrue(shelf.canAccommodateWeight(new BigDecimal("400")));
    }

    @Test
    void canAccommodateWeight_shouldReturnFalseWhenExceedsLimit() {
        shelf.setMaxWeight(new BigDecimal("1000"));
        shelf.setCurrentWeight(new BigDecimal("500"));
        
        assertFalse(shelf.canAccommodateWeight(new BigDecimal("600")));
    }

    @Test
    void addWeight_shouldIncreaseCurrentWeight() {
        shelf.addWeight(new BigDecimal("100"));
        
        assertEquals(new BigDecimal("100"), shelf.getCurrentWeight());
    }

    @Test
    void addWeight_shouldAutoSetFullWhenCapacityReached() {
        shelf.setMaxWeight(new BigDecimal("100"));
        shelf.setCurrentWeight(new BigDecimal("80"));
        
        shelf.addWeight(new BigDecimal("30"));
        
        assertEquals(Shelf.ShelfStatus.FULL, shelf.getStatus());
    }

    @Test
    void removeWeight_shouldDecreaseCurrentWeight() {
        shelf.setCurrentWeight(new BigDecimal("500"));
        
        shelf.removeWeight(new BigDecimal("200"));
        
        assertEquals(new BigDecimal("300"), shelf.getCurrentWeight());
    }

    @Test
    void removeWeight_shouldNotGoBelowZero() {
        shelf.setCurrentWeight(new BigDecimal("100"));
        
        shelf.removeWeight(new BigDecimal("200"));
        
        assertEquals(BigDecimal.ZERO, shelf.getCurrentWeight());
    }

    @Test
    void removeWeight_shouldSetEmptyWhenWeightBecomesZero() {
        shelf.setStatus(Shelf.ShelfStatus.OCCUPIED);
        shelf.setCurrentWeight(new BigDecimal("100"));
        
        shelf.removeWeight(new BigDecimal("100"));
        
        assertEquals(Shelf.ShelfStatus.EMPTY, shelf.getStatus());
    }

    @Test
    void removeWeight_shouldSetOccupiedWhenFullBecomesPartial() {
        shelf.setStatus(Shelf.ShelfStatus.FULL);
        shelf.setMaxWeight(new BigDecimal("1000"));
        shelf.setCurrentWeight(new BigDecimal("1000"));
        
        shelf.removeWeight(new BigDecimal("500"));
        
        assertEquals(Shelf.ShelfStatus.OCCUPIED, shelf.getStatus());
    }

    @Test
    void calculateUtilizationRate_shouldReturnCorrectPercentage() {
        shelf.setMaxWeight(new BigDecimal("1000"));
        shelf.setCurrentWeight(new BigDecimal("250"));
        
        BigDecimal rate = shelf.calculateUtilizationRate();
        
        assertEquals(new BigDecimal("25.00"), rate);
    }

    @Test
    void calculateUtilizationRate_shouldReturnZeroWhenNoMaxWeight() {
        shelf.setMaxWeight(null);
        shelf.setCurrentWeight(new BigDecimal("100"));
        
        assertEquals(BigDecimal.ZERO, shelf.calculateUtilizationRate());
    }
}