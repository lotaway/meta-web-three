package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ProcessRouteTest {
    
    @Test
    void testCreateProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals("PRoute-001", route.getRouteCode());
        assertEquals("产品A工艺路线", route.getRouteName());
        assertEquals("P001", route.getProductCode());
        assertEquals(1, route.getVersion());
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        assertNotNull(route.getCreatedAt());
        assertNotNull(route.getUpdatedAt());
    }
    
    @Test
    void testActivateProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
    
    @Test
    void testArchiveProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        route.activate();
        
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testUpdateVersion() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals(1, route.getVersion());
        route.updateVersion();
        assertEquals(2, route.getVersion());
        route.updateVersion();
        assertEquals(3, route.getVersion());
    }
    
    @Test
    void testProcessRouteFullLifecycle() {
        ProcessRoute route = new ProcessRoute();
        
        route.create("PRoute-001", "产品A工艺路线", "P001");
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        assertEquals(1, route.getVersion());
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        
        route.updateVersion();
        assertEquals(2, route.getVersion());
        
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testProcessRouteWithSteps() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        
        ProcessRoute.ProcessStep step1 = new ProcessRoute.ProcessStep();
        step1.setStepNo(1);
        step1.setProcessCode("PC-001");
        step1.setProcessName("组装");
        step1.setWorkstationId("WS-001");
        step1.setStandardTime(300);
        step1.setQualityCheckpoint("IPQC");
        steps.add(step1);
        
        ProcessRoute.ProcessStep step2 = new ProcessRoute.ProcessStep();
        step2.setStepNo(2);
        step2.setProcessCode("PC-002");
        step2.setProcessName("测试");
        step2.setWorkstationId("WS-002");
        step2.setStandardTime(120);
        step2.setQualityCheckpoint("FQC");
        steps.add(step2);
        
        route.setSteps(steps);
        
        assertEquals(2, route.getSteps().size());
        assertEquals("组装", route.getSteps().get(0).getProcessName());
        assertEquals("测试", route.getSteps().get(1).getProcessName());
    }
    
    @Test
    void testMultipleActivateCalls() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
    
    @Test
    void testArchiveAfterActivateThenArchive() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.activate();
        route.archive();
        
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testArchiveFromDraft() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testActivateFromArchived() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        route.activate();
        route.archive();
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
}