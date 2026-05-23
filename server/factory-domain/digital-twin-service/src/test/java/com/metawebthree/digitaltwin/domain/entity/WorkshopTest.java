package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WorkshopTest {

    @Test
    void create_shouldInitializeWithPlanningStatus() {
        Workshop workshop = new Workshop();
        workshop.create("WS-01", "一号车间", "数字孪生演示车间");
        
        assertEquals("WS-01", workshop.getWorkshopCode());
        assertEquals("一号车间", workshop.getWorkshopName());
        assertEquals("数字孪生演示车间", workshop.getDescription());
        assertEquals(Workshop.WorkshopStatus.PLANNING, workshop.getStatus());
        assertNotNull(workshop.getCreatedAt());
        assertNotNull(workshop.getUpdatedAt());
    }

    @Test
    void startConstruction_shouldChangeStatusToConstruction() {
        Workshop workshop = new Workshop();
        workshop.create("WS-01", "一号车间", "数字孪生演示车间");
        
        workshop.startConstruction();
        
        assertEquals(Workshop.WorkshopStatus.CONSTRUCTION, workshop.getStatus());
    }

    @Test
    void startOperating_shouldChangeStatusToOperating() {
        Workshop workshop = new Workshop();
        workshop.create("WS-01", "一号车间", "数字孪生演示车间");
        
        workshop.startOperating();
        
        assertEquals(Workshop.WorkshopStatus.OPERATING, workshop.getStatus());
    }

    @Test
    void enterMaintenance_shouldChangeStatusToMaintenance() {
        Workshop workshop = new Workshop();
        workshop.create("WS-01", "一号车间", "数字孪生演示车间");
        workshop.startOperating();
        
        workshop.enterMaintenance();
        
        assertEquals(Workshop.WorkshopStatus.MAINTENANCE, workshop.getStatus());
    }

    @Test
    void decommission_shouldChangeStatusToDecommissioned() {
        Workshop workshop = new Workshop();
        workshop.create("WS-01", "一号车间", "数字孪生演示车间");
        
        workshop.decommission();
        
        assertEquals(Workshop.WorkshopStatus.DECOMMISSIONED, workshop.getStatus());
    }

    @Test
    void equals_shouldBeBasedOnWorkshopCode() {
        Workshop workshop1 = new Workshop();
        workshop1.create("WS-01", "一号车间", "描述");
        
        Workshop workshop2 = new Workshop();
        workshop2.create("WS-01", "二号车间", "不同描述");
        
        Workshop workshop3 = new Workshop();
        workshop3.create("WS-02", "三号车间", "描述");
        
        assertEquals(workshop1, workshop2); // Same workshopCode
        assertNotEquals(workshop1, workshop3); // Different workshopCode
    }

    @Test
    void hashCode_shouldBeBasedOnWorkshopCode() {
        Workshop workshop1 = new Workshop();
        workshop1.create("WS-01", "一号车间", "描述");
        
        Workshop workshop2 = new Workshop();
        workshop2.create("WS-01", "二号车间", "不同描述");
        
        assertEquals(workshop1.hashCode(), workshop2.hashCode());
    }
}