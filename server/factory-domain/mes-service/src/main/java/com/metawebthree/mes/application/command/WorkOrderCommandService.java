package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class WorkOrderCommandService {

    private final WorkOrderRepository workOrderRepository;

    public WorkOrderCommandService(WorkOrderRepository workOrderRepository) {
        this.workOrderRepository = workOrderRepository;
    }

    public WorkOrder prepareCreateWorkOrder(String workOrderNo, String productCode, String productName,
                                             Integer quantity, String workshopId, String processRouteId) {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create(workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        return workOrder;
    }

    public void saveWorkOrder(WorkOrder workOrder) {
        workOrderRepository.save(workOrder);
    }

    public WorkOrder prepareCreateWorkOrderWithType(String workOrderNo, String productCode, String productName,
                                                     Integer quantity, String workshopId, String processRouteId,
                                                     String typeCode) {
        WorkOrder workOrder = new WorkOrder();
        workOrder.createWithType(workOrderNo, productCode, productName, quantity, workshopId, processRouteId, typeCode);
        return workOrder;
    }

    public WorkOrder prepareUpdateOrder(Long id, String productCode, String productName, Integer quantity,
                                        String workshopId, String processRouteId, String priority,
                                        LocalDateTime plannedStartTime, LocalDateTime plannedEndTime) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.setProductCode(productCode);
        workOrder.setProductName(productName);
        workOrder.setQuantity(quantity);
        workOrder.setWorkshopId(workshopId);
        workOrder.setProcessRouteId(processRouteId);
        if (priority != null) {
            workOrder.setPriority(WorkOrder.Priority.valueOf(priority));
        }
        workOrder.setPlannedStartTime(plannedStartTime);
        workOrder.setPlannedEndTime(plannedEndTime);
        return workOrder;
    }

    public void saveUpdateOrder(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareRelease(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.release();
        return workOrder;
    }

    public void saveRelease(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareStart(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.start();
        return workOrder;
    }

    public void saveStart(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder preparePause(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.pause();
        return workOrder;
    }

    public void savePause(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareResume(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.resume();
        return workOrder;
    }

    public void saveResume(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareComplete(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.complete();
        return workOrder;
    }

    public void saveComplete(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareCancel(Long id) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.cancel();
        return workOrder;
    }

    public void saveCancel(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public WorkOrder prepareCancelWithReason(Long id, String reason) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.cancelWithReason(reason);
        return workOrder;
    }

    public WorkOrder prepareUpdateProgress(Long id, Integer quantity) {
        WorkOrder workOrder = findWorkOrderOrThrow(id);
        workOrder.updateProgress(quantity);
        return workOrder;
    }

    public void saveUpdateProgress(WorkOrder workOrder) {
        workOrderRepository.update(workOrder);
    }

    public void deleteWorkOrder(Long id) {
        workOrderRepository.deleteById(id);
    }

    private WorkOrder findWorkOrderOrThrow(Long id) {
        return workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
    }
}
