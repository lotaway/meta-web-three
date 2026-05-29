package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class MesDomainServiceImpl implements MesDomainService {

    private final WorkOrderRepository workOrderRepository;
    private final ProductionTaskRepository taskRepository;
    private final ProcessRouteRepository routeRepository;
    private final EquipmentRepository equipmentRepository;
    private final CodeRuleRepository codeRuleRepository;

    public MesDomainServiceImpl(
            WorkOrderRepository workOrderRepository,
            ProductionTaskRepository taskRepository,
            ProcessRouteRepository routeRepository,
            EquipmentRepository equipmentRepository,
            CodeRuleRepository codeRuleRepository) {
        this.workOrderRepository = workOrderRepository;
        this.taskRepository = taskRepository;
        this.routeRepository = routeRepository;
        this.equipmentRepository = equipmentRepository;
        this.codeRuleRepository = codeRuleRepository;
    }

    @Override
    public WorkOrder createWorkOrder(String workOrderNo, String productCode, String productName,
            Integer quantity, String workshopId, String processRouteId) {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create(workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        return workOrderRepository.save(workOrder);
    }

    @Override
    public WorkOrder createWorkOrderWithCodeRule(String businessType, String productCode, String productName,
            Integer quantity, String workshopId, String processRouteId) {
        Optional<CodeRule> codeRuleOpt = codeRuleRepository.findByBusinessTypeAndStatus(
                businessType, CodeRule.RuleStatus.ACTIVE);
        
        String workOrderNo;
        if (codeRuleOpt.isPresent()) {
            CodeRule codeRule = codeRuleOpt.get();
            workOrderNo = codeRule.peekNextCode();
            codeRule.advanceSequence();
            codeRuleRepository.save(codeRule);
        } else {
            throw new IllegalStateException("No active CodeRule found for businessType: " + businessType);
        }
        
        WorkOrder workOrder = new WorkOrder();
        workOrder.create(workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        return workOrderRepository.save(workOrder);
    }

    @Override
    public void releaseWorkOrder(Long workOrderId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId)
            .orElseThrow(() -> new IllegalArgumentException("Work order not found"));
        workOrder.release();
        workOrderRepository.update(workOrder);
    }

    @Override
    public void startWorkOrder(Long workOrderId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId)
            .orElseThrow(() -> new IllegalArgumentException("Work order not found"));
        workOrder.start();
        workOrderRepository.update(workOrder);
    }

    @Override
    public void completeWorkOrder(Long workOrderId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId)
            .orElseThrow(() -> new IllegalArgumentException("Work order not found"));
        workOrder.complete();
        workOrderRepository.update(workOrder);
    }

    @Override
    public void cancelWorkOrder(Long workOrderId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId)
            .orElseThrow(() -> new IllegalArgumentException("Work order not found"));
        workOrder.cancel();
        workOrderRepository.update(workOrder);
    }

    @Override
    public void updateWorkOrderProgress(Long workOrderId, Integer quantity) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId)
            .orElseThrow(() -> new IllegalArgumentException("Work order not found"));
        workOrder.updateProgress(quantity);
        workOrderRepository.update(workOrder);
    }

    @Override
    public ProductionTask createTask(String taskNo, Long workOrderId, String workstationId,
            String processCode, Integer quantity, String operatorId) {
        ProductionTask task = new ProductionTask();
        task.create(taskNo, workOrderId, workstationId, processCode, quantity, operatorId);
        return taskRepository.save(task);
    }

    @Override
    public void startTask(Long taskId) {
        ProductionTask task = taskRepository.findById(taskId)
            .orElseThrow(() -> new IllegalArgumentException("Task not found"));
        task.start();
        taskRepository.update(task);
    }

    @Override
    public void completeTask(Long taskId, Integer qualified, Integer defective) {
        ProductionTask task = taskRepository.findById(taskId)
            .orElseThrow(() -> new IllegalArgumentException("Task not found"));
        task.complete(qualified, defective, 0);
        taskRepository.update(task);
    }

    @Override
    public void passQualityCheck(Long taskId) {
        ProductionTask task = taskRepository.findById(taskId)
            .orElseThrow(() -> new IllegalArgumentException("Task not found"));
        task.passQualityCheck();
        taskRepository.update(task);
    }

    @Override
    public void failQualityCheck(Long taskId) {
        ProductionTask task = taskRepository.findById(taskId)
            .orElseThrow(() -> new IllegalArgumentException("Task not found"));
        task.failQualityCheck();
        taskRepository.update(task);
    }

    @Override
    public ProcessRoute createProcessRoute(String routeCode, String routeName, String productCode) {
        ProcessRoute route = new ProcessRoute();
        route.create(routeCode, routeName, productCode);
        return routeRepository.save(route);
    }

    @Override
    public void activateProcessRoute(Long routeId) {
        ProcessRoute route = routeRepository.findById(routeId)
            .orElseThrow(() -> new IllegalArgumentException("Process route not found"));
        route.activate();
        routeRepository.update(route);
    }

    @Override
    public Equipment createEquipment(String equipmentCode, String equipmentName,
            String equipmentType, String workshopId) {
        Equipment equipment = new Equipment();
        equipment.create(equipmentCode, equipmentName, equipmentType, workshopId);
        return equipmentRepository.save(equipment);
    }

    @Override
    public void startEquipmentTask(Long equipmentId, String taskNo) {
        Equipment equipment = equipmentRepository.findById(equipmentId)
            .orElseThrow(() -> new IllegalArgumentException("Equipment not found"));
        equipment.startTask(taskNo);
        equipmentRepository.update(equipment);
    }

    @Override
    public void completeEquipmentTask(Long equipmentId) {
        Equipment equipment = equipmentRepository.findById(equipmentId)
            .orElseThrow(() -> new IllegalArgumentException("Equipment not found"));
        equipment.completeTask();
        equipmentRepository.update(equipment);
    }

    @Override
    public void reportEquipmentBreakdown(Long equipmentId) {
        Equipment equipment = equipmentRepository.findById(equipmentId)
            .orElseThrow(() -> new IllegalArgumentException("Equipment not found"));
        equipment.reportBreakdown();
        equipmentRepository.update(equipment);
    }

    @Override
    public void repairEquipment(Long equipmentId) {
        Equipment equipment = equipmentRepository.findById(equipmentId)
            .orElseThrow(() -> new IllegalArgumentException("Equipment not found"));
        equipment.repair();
        equipmentRepository.update(equipment);
    }

    @Override
    public List<WorkOrder> getWorkshopWorkOrders(String workshopId) {
        return workOrderRepository.findByWorkshopId(workshopId);
    }

    @Override
    public List<Equipment> getWorkshopEquipment(String workshopId) {
        return equipmentRepository.findByWorkshopId(workshopId);
    }
}