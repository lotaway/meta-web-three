package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ProductionTaskTest {
    
    @Test
    void testCreateProductionTask() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        
        assertEquals("PT-2026-001", task.getTaskNo());
        assertEquals(1L, task.getWorkOrderId());
        assertEquals("WS-001", task.getWorkstationId());
        assertEquals("PC-001", task.getProcessCode());
        assertEquals(100, task.getQuantity());
        assertEquals(0, task.getCompletedQuantity());
        assertEquals(0, task.getQualifiedQuantity());
        assertEquals(0, task.getDefectiveQuantity());
        assertEquals("OP-001", task.getOperatorId());
        assertEquals(ProductionTask.TaskStatus.PENDING, task.getStatus());
        assertNotNull(task.getCreatedAt());
        assertNotNull(task.getUpdatedAt());
    }
    
    @Test
    void testStartPendingTask() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        
        assertEquals(ProductionTask.TaskStatus.PENDING, task.getStatus());
        task.start();
        assertEquals(ProductionTask.TaskStatus.IN_PROGRESS, task.getStatus());
        assertNotNull(task.getStartTime());
    }
    
    @Test
    void testStartNonPendingTaskThrowsException() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        assertThrows(IllegalStateException.class, () -> task.start());
    }
    
    @Test
    void testCompleteInProgressTask() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        task.complete(90, 10);
        assertEquals(ProductionTask.TaskStatus.QUALITY_CHECK, task.getStatus());
        assertEquals(90, task.getQualifiedQuantity());
        assertEquals(10, task.getDefectiveQuantity());
        assertEquals(100, task.getCompletedQuantity());
        assertNotNull(task.getEndTime());
    }
    
    @Test
    void testCompleteNonInProgressTaskThrowsException() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        
        assertThrows(IllegalStateException.class, () -> task.complete(100, 0));
    }
    
    @Test
    void testPassQualityCheck() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        task.complete(90, 10);
        
        task.passQualityCheck();
        assertEquals(ProductionTask.TaskStatus.COMPLETED, task.getStatus());
    }
    
    @Test
    void testPassQualityCheckNotInQualityCheckThrowsException() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        assertThrows(IllegalStateException.class, () -> task.passQualityCheck());
    }
    
    @Test
    void testFailQualityCheck() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        task.complete(90, 10);
        
        task.failQualityCheck();
        assertEquals(ProductionTask.TaskStatus.REWORK, task.getStatus());
    }
    
    @Test
    void testFailQualityCheckNotInQualityCheckThrowsException() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        assertThrows(IllegalStateException.class, () -> task.failQualityCheck());
    }
    
    @Test
    void testScrap() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        task.scrap();
        assertEquals(ProductionTask.TaskStatus.SCRAP, task.getStatus());
    }
    
    @Test
    void testFullTaskLifecycle() {
        ProductionTask task = new ProductionTask();
        
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        assertEquals(ProductionTask.TaskStatus.PENDING, task.getStatus());
        
        task.start();
        assertEquals(ProductionTask.TaskStatus.IN_PROGRESS, task.getStatus());
        assertNotNull(task.getStartTime());
        
        task.complete(95, 5);
        assertEquals(ProductionTask.TaskStatus.QUALITY_CHECK, task.getStatus());
        assertEquals(95, task.getQualifiedQuantity());
        assertEquals(5, task.getDefectiveQuantity());
        assertEquals(100, task.getCompletedQuantity());
        
        task.passQualityCheck();
        assertEquals(ProductionTask.TaskStatus.COMPLETED, task.getStatus());
        assertNotNull(task.getEndTime());
    }
    
    @Test
    void testTaskLifecycleWithRework() {
        ProductionTask task = new ProductionTask();
        
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        task.complete(80, 20);
        task.failQualityCheck();
        
        assertEquals(ProductionTask.TaskStatus.REWORK, task.getStatus());
    }
    
    @Test
    void testCompleteWithZeroQualified() {
        ProductionTask task = new ProductionTask();
        task.create("PT-2026-001", 1L, "WS-001", "PC-001", 100, "OP-001");
        task.start();
        
        task.complete(0, 100);
        assertEquals(0, task.getQualifiedQuantity());
        assertEquals(100, task.getDefectiveQuantity());
        assertEquals(100, task.getCompletedQuantity());
    }
}