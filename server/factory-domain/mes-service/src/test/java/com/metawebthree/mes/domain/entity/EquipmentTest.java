package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EquipmentTest {
    
    @Test
    void createShouldInitializeEquipmentWithDefaultValues() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        assertEquals("EQ001", equipment.getEquipmentCode());
        assertEquals("CNC机床1", equipment.getEquipmentName());
        assertEquals("CNC", equipment.getEquipmentType());
        assertEquals("WS001", equipment.getWorkshopId());
        assertEquals(Equipment.EquipmentStatus.IDLE, equipment.getStatus());
        assertEquals(0.0, equipment.getUtilizationRate());
        assertEquals(0, equipment.getTodayOutput());
        assertNotNull(equipment.getCreatedAt());
    }
    
    @Test
    void startTaskShouldSetEquipmentToRunning() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        equipment.startTask("WO20260101");
        
        assertEquals(Equipment.EquipmentStatus.RUNNING, equipment.getStatus());
        assertEquals("WO20260101", equipment.getCurrentTaskNo());
    }
    
    @Test
    void startTaskShouldThrowExceptionWhenNotIdle() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.startTask("WO20260102");
        });
    }
    
    @Test
    void completeTaskShouldResetEquipmentToIdle() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        
        equipment.completeTask();
        
        assertEquals(Equipment.EquipmentStatus.IDLE, equipment.getStatus());
        assertNull(equipment.getCurrentTaskNo());
        assertEquals(1, equipment.getTodayOutput());
    }
    
    @Test
    void completeTaskShouldThrowExceptionWhenNotRunning() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.completeTask();
        });
    }
    
    @Test
    void reportBreakdownShouldSetStatusToBreakdown() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        
        equipment.reportBreakdown();
        
        assertEquals(Equipment.EquipmentStatus.BREAKDOWN, equipment.getStatus());
    }
    
    @Test
    void reportBreakdownShouldThrowExceptionWhenNotRunning() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.reportBreakdown();
        });
    }
    
    @Test
    void repairShouldResetEquipmentToIdle() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        equipment.reportBreakdown();
        
        equipment.repair();
        
        assertEquals(Equipment.EquipmentStatus.IDLE, equipment.getStatus());
    }
    
    @Test
    void repairShouldThrowExceptionWhenNotBrokenDown() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.repair();
        });
    }
    
    @Test
    void startMaintenanceShouldSetStatusToMaintenance() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        equipment.startMaintenance();
        
        assertEquals(Equipment.EquipmentStatus.MAINTENANCE, equipment.getStatus());
    }
    
    @Test
    void startMaintenanceShouldThrowExceptionWhenRunning() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.startMaintenance();
        });
    }
    
    @Test
    void completeMaintenanceShouldResetToIdleAndUpdateMaintenanceTime() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startMaintenance();
        
        equipment.completeMaintenance();
        
        assertEquals(Equipment.EquipmentStatus.IDLE, equipment.getStatus());
        assertNotNull(equipment.getLastMaintenanceTime());
    }
    
    @Test
    void completeMaintenanceShouldThrowExceptionWhenNotInMaintenance() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        assertThrows(IllegalStateException.class, () -> {
            equipment.completeMaintenance();
        });
    }
    
    @Test
    void multipleTaskCyclesShouldIncrementTodayOutput() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        
        equipment.startTask("WO20260101");
        equipment.completeTask();
        equipment.startTask("WO20260102");
        equipment.completeTask();
        equipment.startTask("WO20260103");
        equipment.completeTask();
        
        assertEquals(3, equipment.getTodayOutput());
    }
    
    @Test
    void equipmentCanTransitionFromBreakdownToMaintenance() {
        Equipment equipment = new Equipment();
        equipment.create("EQ001", "CNC机床1", "CNC", "WS001");
        equipment.startTask("WO20260101");
        equipment.reportBreakdown();
        
        equipment.repair();
        equipment.startMaintenance();
        
        assertEquals(Equipment.EquipmentStatus.MAINTENANCE, equipment.getStatus());
    }
}