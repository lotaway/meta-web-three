package com.metawebthree.cs.domain.repository;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
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

    // Paged queries
    IPage<WorkOrder> findByCustomerIdPaged(Page<WorkOrder> page, Long customerId);
    IPage<WorkOrder> findByAgentIdPaged(Page<WorkOrder> page, Long agentId);
    IPage<WorkOrder> findByStatusPaged(Page<WorkOrder> page, WorkOrderStatus status);
    IPage<WorkOrder> findByCategoryPaged(Page<WorkOrder> page, WorkOrderCategory category);
    IPage<WorkOrder> findPendingPaged(Page<WorkOrder> page);
}