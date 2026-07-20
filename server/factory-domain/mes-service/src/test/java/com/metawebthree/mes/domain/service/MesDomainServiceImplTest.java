package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MesDomainServiceImplTest {

    @Mock
    private WorkOrderRepository workOrderRepository;
    @Mock
    private ProductionTaskRepository taskRepository;
    @Mock
    private ProcessRouteRepository routeRepository;
    @Mock
    private EquipmentRepository equipmentRepository;
    @Mock
    private CodeRuleRepository codeRuleRepository;

    @Captor
    private ArgumentCaptor<WorkOrder> workOrderCaptor;
    @Captor
    private ArgumentCaptor<ProductionTask> taskCaptor;

    private MesDomainServiceImpl mesDomainService;

    @BeforeEach
    void setUp() {
        mesDomainService = new MesDomainServiceImpl(
                workOrderRepository, taskRepository,
                routeRepository, equipmentRepository, codeRuleRepository);
    }

    @Test
    void createWorkOrder_shouldSaveAndReturnWorkOrder() {
        WorkOrder saved = new WorkOrder();
        saved.setId(1L);
        saved.setWorkOrderNo("WO-001");
        when(workOrderRepository.save(any(WorkOrder.class))).thenReturn(saved);

        WorkOrder result = mesDomainService.createWorkOrder(
                "WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");

        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals("WO-001", result.getWorkOrderNo());
        verify(workOrderRepository).save(workOrderCaptor.capture());
        WorkOrder captured = workOrderCaptor.getValue();
        assertEquals("WO-001", captured.getWorkOrderNo());
        assertEquals("P001", captured.getProductCode());
        assertEquals("ProductA", captured.getProductName());
        assertEquals(100, captured.getQuantity());
        assertEquals("WS-001", captured.getWorkshopId());
        assertEquals("PR-001", captured.getProcessRouteId());
    }

    @Test
    void releaseWorkOrder_shouldFindUpdateAndSave() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));

        mesDomainService.releaseWorkOrder(1L);

        assertEquals(WorkOrder.WorkOrderStatus.RELEASED, workOrder.getStatus());
        verify(workOrderRepository).findById(1L);
        verify(workOrderRepository).update(workOrder);
    }

    @Test
    void startWorkOrder_shouldFindUpdateAndSave() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");
        workOrder.release();
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));

        mesDomainService.startWorkOrder(1L);

        assertEquals(WorkOrder.WorkOrderStatus.IN_PROGRESS, workOrder.getStatus());
        verify(workOrderRepository).findById(1L);
        verify(workOrderRepository).update(workOrder);
    }

    @Test
    void completeWorkOrder_shouldFindUpdateAndSave() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        workOrder.updateProgress(100);
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));

        mesDomainService.completeWorkOrder(1L);

        assertEquals(WorkOrder.WorkOrderStatus.COMPLETED, workOrder.getStatus());
        verify(workOrderRepository).findById(1L);
        verify(workOrderRepository).update(workOrder);
    }

    @Test
    void cancelWorkOrder_shouldFindUpdateAndSave() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");
        workOrder.release();
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));

        mesDomainService.cancelWorkOrder(1L);

        assertEquals(WorkOrder.WorkOrderStatus.CANCELLED, workOrder.getStatus());
        verify(workOrderRepository).findById(1L);
        verify(workOrderRepository).update(workOrder);
    }

    @Test
    void updateWorkOrderProgress_shouldFindUpdateAndSave() {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create("WO-001", "P001", "ProductA", 100, "WS-001", "PR-001");
        workOrder.release();
        workOrder.start();
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));

        mesDomainService.updateWorkOrderProgress(1L, 30);

        assertEquals(30, workOrder.getCompletedQuantity());
        verify(workOrderRepository).findById(1L);
        verify(workOrderRepository).update(workOrder);
    }

    @Test
    void workOrderLifecycleMethod_whenNotFound_shouldThrowIllegalArgumentException() {
        when(workOrderRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.releaseWorkOrder(99L));
        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.startWorkOrder(99L));
        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.completeWorkOrder(99L));
        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.cancelWorkOrder(99L));
        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.updateWorkOrderProgress(99L, 10));

        verify(workOrderRepository, times(5)).findById(99L);
        verify(workOrderRepository, never()).update(any());
    }

    @Test
    void createWorkOrderWithCodeRule_whenCodeRuleExists_shouldGenerateCodeAndSave() {
        CodeRule codeRule = CodeRule.create("RULE-001", "WO Rule", "WORK_ORDER", "{PREFIX}{SEQ}", 5);
        codeRule.setId(1L);
        when(codeRuleRepository.findByBusinessTypeAndStatus("WORK_ORDER", CodeRule.RuleStatus.ACTIVE))
                .thenReturn(Optional.of(codeRule));
        WorkOrder saved = new WorkOrder();
        saved.setId(1L);
        saved.setWorkOrderNo(codeRule.peekNextCode() + "001");
        when(workOrderRepository.save(any(WorkOrder.class))).thenReturn(saved);

        WorkOrder result = mesDomainService.createWorkOrderWithCodeRule(
                "WORK_ORDER", "P001", "ProductA", 100, "WS-001", "PR-001");

        assertNotNull(result);
        assertEquals(2L, codeRule.getCurrentValue().longValue());
        verify(codeRuleRepository).findByBusinessTypeAndStatus("WORK_ORDER", CodeRule.RuleStatus.ACTIVE);
        verify(codeRuleRepository).save(codeRule);
        verify(workOrderRepository).save(any(WorkOrder.class));
    }

    @Test
    void createWorkOrderWithCodeRule_whenNoCodeRule_shouldThrowIllegalStateException() {
        when(codeRuleRepository.findByBusinessTypeAndStatus("WORK_ORDER", CodeRule.RuleStatus.ACTIVE))
                .thenReturn(Optional.empty());

        assertThrows(IllegalStateException.class,
                () -> mesDomainService.createWorkOrderWithCodeRule(
                        "WORK_ORDER", "P001", "ProductA", 100, "WS-001", "PR-001"));

        verify(codeRuleRepository).findByBusinessTypeAndStatus("WORK_ORDER", CodeRule.RuleStatus.ACTIVE);
        verifyNoInteractions(workOrderRepository);
    }

    @Test
    void createTask_shouldSaveAndReturnTask() {
        ProductionTask saved = new ProductionTask();
        saved.setId(1L);
        saved.setTaskNo("TASK-001");
        when(taskRepository.save(any(ProductionTask.class))).thenReturn(saved);

        ProductionTask result = mesDomainService.createTask(
                "TASK-001", 1L, "10", "PC-001", 50, "OP-001");

        assertNotNull(result);
        assertEquals(1L, result.getId());
        verify(taskRepository).save(taskCaptor.capture());
        ProductionTask captured = taskCaptor.getValue();
        assertEquals("TASK-001", captured.getTaskNo());
        assertEquals(1L, captured.getWorkOrderId());
        assertEquals(10L, captured.getWorkstationId());
        assertEquals("PC-001", captured.getProcessCode());
        assertEquals(50, captured.getQuantity());
        assertEquals("OP-001", captured.getOperatorId());
        assertEquals(ProductionTask.TaskStatus.PENDING, captured.getStatus());
    }

    @Test
    void startTask_shouldFindUpdateAndSave() {
        ProductionTask task = new ProductionTask();
        task.create("TASK-001", 1L, 10L, "PC-001", 50, "OP-001");
        when(taskRepository.findById(1L)).thenReturn(Optional.of(task));

        mesDomainService.startTask(1L);

        assertEquals(ProductionTask.TaskStatus.IN_PROGRESS, task.getStatus());
        verify(taskRepository).findById(1L);
        verify(taskRepository).update(task);
    }

    @Test
    void completeTask_shouldFindUpdateAndSave() {
        ProductionTask task = new ProductionTask();
        task.create("TASK-001", 1L, 10L, "PC-001", 50, "OP-001");
        task.start();
        when(taskRepository.findById(1L)).thenReturn(Optional.of(task));

        mesDomainService.completeTask(1L, 45, 5);

        assertEquals(ProductionTask.TaskStatus.QUALITY_CHECK, task.getStatus());
        assertEquals(45, task.getQualifiedQuantity());
        assertEquals(5, task.getDefectiveQuantity());
        assertNotNull(task.getEndTime());
        verify(taskRepository).findById(1L);
        verify(taskRepository).update(task);
    }

    @Test
    void taskMethod_whenNotFound_shouldThrowIllegalArgumentException() {
        when(taskRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.startTask(99L));
        assertThrows(IllegalArgumentException.class,
                () -> mesDomainService.completeTask(99L, 0, 0));

        verify(taskRepository, times(2)).findById(99L);
        verify(taskRepository, never()).update(any());
    }
}
