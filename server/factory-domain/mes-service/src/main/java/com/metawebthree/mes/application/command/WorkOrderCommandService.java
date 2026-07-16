package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.infrastructure.event.MesCrossDomainEventPublisher;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class WorkOrderCommandService {
    
    private final WorkOrderRepository workOrderRepository;
    private final MesCrossDomainEventPublisher crossDomainEventPublisher;
    
    public WorkOrderCommandService(WorkOrderRepository workOrderRepository,
                                   MesCrossDomainEventPublisher crossDomainEventPublisher) {
        this.workOrderRepository = workOrderRepository;
        this.crossDomainEventPublisher = crossDomainEventPublisher;
    }
    
    public WorkOrder createWorkOrder(String workOrderNo, String productCode, String productName,
                                     Integer quantity, String workshopId, String processRouteId) {
        WorkOrder workOrder = new WorkOrder();
        workOrder.create(workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        return workOrderRepository.save(workOrder);
    }
    
    public WorkOrder createWorkOrderWithType(String workOrderNo, String productCode, String productName,
                                             Integer quantity, String workshopId, String processRouteId,
                                             String typeCode) {
        WorkOrder workOrder = new WorkOrder();
        workOrder.createWithType(workOrderNo, productCode, productName, quantity, workshopId, processRouteId, typeCode);
        return workOrderRepository.save(workOrder);
    }
    
    public WorkOrder updateWorkOrder(Long id, String productCode, String productName, Integer quantity,
                                     String workshopId, String processRouteId, String priority,
                                     LocalDateTime plannedStartTime, LocalDateTime plannedEndTime) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        
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
        
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder releaseWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.release();
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder startWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.start();
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder pauseWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.pause();
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder resumeWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.resume();
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder completeWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.complete();
        workOrderRepository.update(workOrder);
        crossDomainEventPublisher.publishWorkOrderCompleted(
                workOrder.getId(), workOrder.getWorkOrderNo(),
                workOrder.getProductCode(), workOrder.getQuantity());
        return workOrder;
    }
    
    public WorkOrder cancelWorkOrder(Long id) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.cancel();
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder cancelWorkOrderWithReason(Long id, String reason) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.cancelWithReason(reason);
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public WorkOrder updateProgress(Long id, Integer quantity) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        workOrder.updateProgress(quantity);
        workOrderRepository.update(workOrder);
        return workOrder;
    }
    
    public void deleteWorkOrder(Long id) {
        workOrderRepository.deleteById(id);
    }
    
    public List<WorkOrder> splitWorkOrder(Long id, String splitType, Integer splitCount) {
        WorkOrder parentOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Work order not found: " + id));
        
        if (parentOrder.getQuantity() == null || parentOrder.getQuantity() <= 0) {
            throw new IllegalArgumentException("Invalid quantity for split");
        }
        
        int quantityPerChild = parentOrder.getQuantity() / splitCount;
        int remainder = parentOrder.getQuantity() % splitCount;
        
        java.util.ArrayList<WorkOrder> childOrders = new java.util.ArrayList<>();
        
        for (int i = 1; i <= splitCount; i++) {
            WorkOrder child = new WorkOrder();
            String childWorkOrderNo = parentOrder.getWorkOrderNo() + "-S" + i;
            int childQuantity = quantityPerChild + (i <= remainder ? 1 : 0);
            
            child.createWithType(childWorkOrderNo, parentOrder.getProductCode(), 
                    parentOrder.getProductName(), childQuantity, 
                    parentOrder.getWorkshopId(), parentOrder.getProcessRouteId(),
                    parentOrder.getTypeCode());
            
            child.setParentWorkOrderId(parentOrder.getId());
            child.setSplitRuleId(parentOrder.getSplitRuleId());
            child.setSplitSequence(i);
            child.setSplitType(splitType);
            child.setPriority(parentOrder.getPriority());
            child.setPlannedStartTime(parentOrder.getPlannedStartTime());
            child.setPlannedEndTime(parentOrder.getPlannedEndTime());
            
            workOrderRepository.save(child);
            childOrders.add(child);
        }
        
        return childOrders;
    }
}