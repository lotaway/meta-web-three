package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.WorkOrder;
import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;
import com.metawebthree.cs.domain.repository.WorkOrderRepository;

import java.util.List;
import java.util.Optional;

public class WorkOrderService {
    private final WorkOrderRepository workOrderRepository;
    private final TicketClassificationService classificationService;

    public WorkOrderService(WorkOrderRepository workOrderRepository, 
                           TicketClassificationService classificationService) {
        this.workOrderRepository = workOrderRepository;
        this.classificationService = classificationService;
    }

    public WorkOrder createWorkOrder(Long customerId, String title, String description) {
        WorkOrder workOrder = new WorkOrder(customerId, title, description, null);
        
        WorkOrderCategory suggestedCategory = classificationService.classify(title, description);
        Double confidence = classificationService.getConfidence();
        
        workOrder.updateCategory(suggestedCategory, confidence);
        
        return workOrderRepository.save(workOrder);
    }

    public WorkOrder createWorkOrder(Long customerId, String title, String description, 
                                    WorkOrderCategory category) {
        WorkOrder workOrder = new WorkOrder(customerId, title, description, category);
        return workOrderRepository.save(workOrder);
    }

    public Optional<WorkOrder> getWorkOrder(Long id) {
        return Optional.ofNullable(workOrderRepository.findById(id));
    }

    public List<WorkOrder> getCustomerWorkOrders(Long customerId) {
        return workOrderRepository.findByCustomerId(customerId);
    }

    public List<WorkOrder> getAgentWorkOrders(Long agentId) {
        return workOrderRepository.findByAgentId(agentId);
    }

    public WorkOrder assignAgent(Long workOrderId, Long agentId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId);
        if (workOrder != null) {
            workOrder.assign(agentId);
            return workOrderRepository.save(workOrder);
        }
        return null;
    }

    public WorkOrder resolveWorkOrder(Long workOrderId, String resolution) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId);
        if (workOrder != null) {
            workOrder.resolve(resolution);
            return workOrderRepository.save(workOrder);
        }
        return null;
    }

    public WorkOrder escalateWorkOrder(Long workOrderId) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId);
        if (workOrder != null) {
            workOrder.escalate();
            return workOrderRepository.save(workOrder);
        }
        return null;
    }

    public WorkOrder reclassify(Long workOrderId, String title, String description) {
        WorkOrder workOrder = workOrderRepository.findById(workOrderId);
        if (workOrder != null) {
            WorkOrderCategory newCategory = classificationService.classify(title, description);
            Double confidence = classificationService.getConfidence();
            workOrder.updateCategory(newCategory, confidence);
            return workOrderRepository.save(workOrder);
        }
        return null;
    }

    public List<WorkOrder> getPendingWorkOrders() {
        return workOrderRepository.findPending();
    }

    public List<WorkOrder> getWorkOrdersByStatus(WorkOrderStatus status) {
        return workOrderRepository.findByStatus(status);
    }

    public List<WorkOrder> getWorkOrdersByCategory(WorkOrderCategory category) {
        return workOrderRepository.findByCategory(category);
    }

    public Long countByStatus(WorkOrderStatus status) {
        return workOrderRepository.countByStatus(status);
    }

    public Long countByCategory(WorkOrderCategory category) {
        return workOrderRepository.countByCategory(category);
    }
}