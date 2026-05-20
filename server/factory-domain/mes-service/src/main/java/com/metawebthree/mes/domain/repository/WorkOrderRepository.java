package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.WorkOrder;
import java.util.List;
import java.util.Optional;

public interface WorkOrderRepository {
    Optional<WorkOrder> findById(Long id);
    Optional<WorkOrder> findByWorkOrderNo(String workOrderNo);
    List<WorkOrder> findByStatus(WorkOrder.WorkOrderStatus status);
    List<WorkOrder> findByWorkshopId(String workshopId);
    List<WorkOrder> findByProductCode(String productCode);
    WorkOrder save(WorkOrder workOrder);
    void update(WorkOrder workOrder);
    void deleteById(Long id);
}