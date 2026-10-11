package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.service.MesDomainService;
import com.metawebthree.mes.infrastructure.event.MesEventPublisher;
import org.springframework.stereotype.Service;

@Service
public class MesCommandService {

    private final MesDomainService domainService;
    private final MesEventPublisher eventPublisher;

    public MesCommandService(MesDomainService domainService, MesEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
    }

    public Long createWorkOrder(String workOrderNo, String productCode, String productName,
                               Integer quantity, String workshopId, String processRouteId) {
        var workOrder = domainService.createWorkOrder(
            workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        eventPublisher.publishWorkOrderCreated(workOrder.getId(), workOrderNo);
        return workOrder.getId();
    }

    public void releaseWorkOrder(Long workOrderId) {
        domainService.releaseWorkOrder(workOrderId);
        eventPublisher.publishWorkOrderReleased(workOrderId);
    }

    public void startWorkOrder(Long workOrderId) {
        domainService.startWorkOrder(workOrderId);
        eventPublisher.publishWorkOrderStarted(workOrderId);
    }

    public void completeWorkOrder(Long workOrderId) {
        domainService.completeWorkOrder(workOrderId);
        eventPublisher.publishWorkOrderCompleted(workOrderId);
    }

    public void createTask(String taskNo, Long workOrderId, String workstationId,
                          String processCode, Integer quantity, String operatorId) {
        var task = domainService.createTask(
            taskNo, workOrderId, workstationId, processCode, quantity, operatorId);
        eventPublisher.publishTaskCreated(task.getId(), taskNo);
    }

    public void startTask(Long taskId) {
        domainService.startTask(taskId);
        eventPublisher.publishTaskStarted(taskId);
    }

    public void completeTask(Long taskId, Integer qualified, Integer defective) {
        domainService.completeTask(taskId, qualified, defective);
        eventPublisher.publishTaskCompleted(taskId, qualified, defective);
    }

    public void reportEquipmentBreakdown(Long equipmentId) {
        domainService.reportEquipmentBreakdown(equipmentId);
        eventPublisher.publishEquipmentBreakdown(equipmentId);
    }

    public void repairEquipment(Long equipmentId) {
        domainService.repairEquipment(equipmentId);
        eventPublisher.publishEquipmentRepaired(equipmentId);
    }
}