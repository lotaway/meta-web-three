package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.WorkOrder;
import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;

import java.util.List;

public interface WorkOrderRepository {
    WorkOrder save(WorkOrder workOrder);
    
    WorkOrder findById(Long id);
    
    List<WorkOrder> findByCustomerId(Long customerId);
    
    List<WorkOrder> findByAgentId(Long agentId);
    
    List<WorkOrder> findByStatus(WorkOrderStatus status);
    
    List<WorkOrder> findByCategory(WorkOrderCategory category);
    
    List<WorkOrder> findPending();
    
    void deleteById(Long id);
    
    List<WorkOrder> findAll();
    
    Long countByStatus(WorkOrderStatus status);
    
    Long countByCategory(WorkOrderCategory category);
}