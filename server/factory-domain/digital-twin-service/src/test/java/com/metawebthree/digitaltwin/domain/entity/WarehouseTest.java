package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

class WarehouseTest {

    private Warehouse warehouse;

    @BeforeEach
    void setUp() {
        warehouse = new Warehouse("WH-001", "Main Warehouse");
    }

    @Test
    void constructor_shouldInitializeWithPlanningStatus() {
        assertEquals("WH-001", warehouse.getWarehouseCode());
        assertEquals("Main Warehouse", warehouse.getWarehouseName());
        assertEquals(Warehouse.WarehouseStatus.PLANNING, warehouse.getStatus());
        assertNotNull(warehouse.getCreatedAt());
        assertNotNull(warehouse.getUpdatedAt());
    }

    @Test
    void activate_shouldSetStatusToOperating() {
        warehouse.activate();
        
        assertEquals(Warehouse.WarehouseStatus.OPERATING, warehouse.getStatus());
        assertNotNull(warehouse.getUpdatedAt());
    }

    @Test
    void enterMaintenance_shouldSetStatusToMaintenance() {
        warehouse.activate();
        warehouse.enterMaintenance();
        
        assertEquals(Warehouse.WarehouseStatus.MAINTENANCE, warehouse.getStatus());
    }

    @Test
    void decommission_shouldSetStatusToDecommissioned() {
        warehouse.activate();
        warehouse.decommission();
        
        assertEquals(Warehouse.WarehouseStatus.DECOMMISSIONED, warehouse.getStatus());
    }

    @Test
    void calculateUtilizationRate_shouldReturnZeroWhenTotalAreaIsNull() {
        warehouse.setTotalArea(null);
        
        assertEquals(BigDecimal.ZERO, warehouse.calculateUtilizationRate());
    }

    @Test
    void calculateUtilizationRate_shouldReturnZeroWhenTotalAreaIsZero() {
        warehouse.setTotalArea(BigDecimal.ZERO);
        
        assertEquals(BigDecimal.ZERO, warehouse.calculateUtilizationRate());
    }

    @Test
    void calculateUtilizationRate_shouldCalculateCorrectly() {
        warehouse.setTotalArea(new BigDecimal("1000"));
        warehouse.setUsedArea(new BigDecimal("250"));
        
        BigDecimal rate = warehouse.calculateUtilizationRate();
        
        assertEquals(new BigDecimal("25.00"), rate);
    }

    @Test
    void calculateUtilizationRate_shouldHandleFullUtilization() {
        warehouse.setTotalArea(new BigDecimal("1000"));
        warehouse.setUsedArea(new BigDecimal("1000"));
        
        assertEquals(new BigDecimal("100.00"), warehouse.calculateUtilizationRate());
    }

    @Test
    void calculateUtilizationRate_shouldRoundToTwoDecimalPlaces() {
        warehouse.setTotalArea(new BigDecimal("1000"));
        warehouse.setUsedArea(new BigDecimal("333"));
        
        BigDecimal rate = warehouse.calculateUtilizationRate();
        
        assertEquals(new BigDecimal("33.30"), rate);
    }

    @Test
    void updateArea_shouldUpdateBothFields() {
        warehouse.updateArea(new BigDecimal("2000"), new BigDecimal("500"));
        
        assertEquals(new BigDecimal("2000"), warehouse.getTotalArea());
        assertEquals(new BigDecimal("500"), warehouse.getUsedArea());
    }

    @Test
    void updateArea_shouldRecalculateUtilization() {
        warehouse.setTotalArea(new BigDecimal("1000"));
        warehouse.setUsedArea(new BigDecimal("250"));
        
        warehouse.updateArea(new BigDecimal("2000"), new BigDecimal("1000"));
        
        assertEquals(new BigDecimal("50.00"), warehouse.calculateUtilizationRate());
    }
}