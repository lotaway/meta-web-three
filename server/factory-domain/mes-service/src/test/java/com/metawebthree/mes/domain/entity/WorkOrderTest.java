package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.*;

public class WorkOrderTest {
    
    @Test
    void testCreateWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertEquals("WO-2026-001", workOrder.getWorkOrderNo());
        assertEquals("P001", workOrder.getProductCode());
        assertEquals("产品A", workOrder.getProductName());
        assertEquals(100, workOrder.getQuantity());
        assertEquals(0, workOrder.getCompletedQuantity());
        assertEquals(WorkOrder.WorkOrderStatus.DRAFT, workOrder.getStatus());
        assertEquals(WorkOrder.Priority.NORMAL, workOrder.getPriority());
        assertEquals("WS-001", workOrder.getWorkshopId());
        assertEquals("PR-001", workOrder.getProcessRouteId());
        assertNotNull(workOrder.getCreatedAt());
        assertNotNull(workOrder.getUpdatedAt());
    }
    
    @Test
    void testReleaseWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertEquals(WorkOrder.WorkOrderStatus.DRAFT, workOrder.getStatus());
        workOrder.release();
        assertEquals(WorkOrder.WorkOrderStatus.RELEASED, workOrder.getStatus());
    }
    
    @Test
    void testReleaseNonDraftWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        
        assertThrows(IllegalStateException.class, () -> workOrder.release());
    }
    
    @Test
    void testStartWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        
        assertEquals(WorkOrder.WorkOrderStatus.IN_PROGRESS, workOrder.getStatus());
        assertNotNull(workOrder.getActualStartTime());
    }
    
    @Test
    void testStartNonReleasedWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertThrows(IllegalStateException.class, () -> workOrder.start());
    }
    
    @Test
    void testPauseWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.pause();
        
        assertEquals(WorkOrder.WorkOrderStatus.PAUSED, workOrder.getStatus());
    }
    
    @Test
    void testPauseNonInProgressWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertThrows(IllegalStateException.class, () -> workOrder.pause());
    }
    
    @Test
    void testResumeWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.pause();
        workOrder.resume();
        
        assertEquals(WorkOrder.WorkOrderStatus.IN_PROGRESS, workOrder.getStatus());
    }
    
    @Test
    void testResumeNonPausedWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertThrows(IllegalStateException.class, () -> workOrder.resume());
    }
    
    @Test
    void testCompleteWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.updateProgress(100);
        workOrder.complete();
        
        assertEquals(WorkOrder.WorkOrderStatus.COMPLETED, workOrder.getStatus());
        assertNotNull(workOrder.getActualEndTime());
    }
    
    @Test
    void testCompleteWorkOrderWithInsufficientQuantityThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.updateProgress(50);
        
        assertThrows(IllegalStateException.class, () -> workOrder.complete());
    }
    
    @Test
    void testCompleteNonInProgressWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertThrows(IllegalStateException.class, () -> workOrder.complete());
    }
    
    @Test
    void testCancelWorkOrder() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.cancel();
        
        assertEquals(WorkOrder.WorkOrderStatus.CANCELLED, workOrder.getStatus());
    }
    
    @Test
    void testCancelCompletedWorkOrderThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.updateProgress(100);
        workOrder.complete();
        
        assertThrows(IllegalStateException.class, () -> workOrder.cancel());
    }
    
    @Test
    void testUpdateProgress() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        
        assertEquals(0, workOrder.getCompletedQuantity());
        workOrder.updateProgress(30);
        assertEquals(30, workOrder.getCompletedQuantity());
        workOrder.updateProgress(50);
        assertEquals(80, workOrder.getCompletedQuantity());
    }
    
    @Test
    void testUpdateProgressNotInProgressThrowsException() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        
        assertThrows(IllegalStateException.class, () -> workOrder.updateProgress(50));
    }
    
    @Test
    void testUpdateProgressExceedsQuantity() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        
        workOrder.updateProgress(120);
        assertEquals(100, workOrder.getCompletedQuantity());
    }
    
    @Test
    void testCompletionRate() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        
        assertEquals(0.0, workOrder.getCompletionRate());
        
        workOrder.updateProgress(25);
        assertEquals(25.0, workOrder.getCompletionRate());
        
        workOrder.updateProgress(25);
        assertEquals(50.0, workOrder.getCompletionRate());
        
        workOrder.updateProgress(50);
        assertEquals(100.0, workOrder.getCompletionRate());
    }
    
    @Test
    void testCompletionRateWithZeroQuantity() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-2026-001", "P001", "产品A", 0, "WS-001", "PR-001");
        
        assertEquals(0.0, workOrder.getCompletionRate());
    }
    
    @Test
    void testFullWorkOrderLifecycle() {
        WorkOrder workOrder = new WorkOrder();
        
        workOrder.create("WO-2026-001", "P001", "产品A", 100, "WS-001", "PR-001");
        assertEquals(WorkOrder.WorkOrderStatus.DRAFT, workOrder.getStatus());
        
        workOrder.release();
        assertEquals(WorkOrder.WorkOrderStatus.RELEASED, workOrder.getStatus());
        
        workOrder.start();
        assertEquals(WorkOrder.WorkOrderStatus.IN_PROGRESS, workOrder.getStatus());
        assertNotNull(workOrder.getActualStartTime());
        
        workOrder.updateProgress(40);
        assertEquals(40, workOrder.getCompletedQuantity());
        assertEquals(40.0, workOrder.getCompletionRate());
        
        workOrder.pause();
        assertEquals(WorkOrder.WorkOrderStatus.PAUSED, workOrder.getStatus());
        
        workOrder.resume();
        assertEquals(WorkOrder.WorkOrderStatus.IN_PROGRESS, workOrder.getStatus());
        
        workOrder.updateProgress(60);
        workOrder.complete();
        assertEquals(WorkOrder.WorkOrderStatus.COMPLETED, workOrder.getStatus());
        assertNotNull(workOrder.getActualEndTime());
    }
}