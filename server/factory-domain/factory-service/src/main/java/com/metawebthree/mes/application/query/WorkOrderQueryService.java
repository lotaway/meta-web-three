package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class WorkOrderQueryService {
    
    private final WorkOrderRepository workOrderRepository;
    
    public WorkOrderQueryService(WorkOrderRepository workOrderRepository) {
        this.workOrderRepository = workOrderRepository;
    }
    
    public Optional<WorkOrder> findById(Long id) {
        return workOrderRepository.findById(id);
    }
    
    public Optional<WorkOrder> findByWorkOrderNo(String workOrderNo) {
        return workOrderRepository.findByWorkOrderNo(workOrderNo);
    }
    
    public List<WorkOrder> findByStatus(WorkOrder.WorkOrderStatus status) {
        return workOrderRepository.findByStatus(status);
    }
    
    public List<WorkOrder> findByWorkshopId(String workshopId) {
        return workOrderRepository.findByWorkshopId(workshopId);
    }
    
    public List<WorkOrder> findByProductCode(String productCode) {
        return workOrderRepository.findByProductCode(productCode);
    }
    
    public List<WorkOrder> findByParentWorkOrderId(Long parentWorkOrderId) {
        return workOrderRepository.findByParentWorkOrderId(parentWorkOrderId);
    }
    
    public List<WorkOrder> findAll() {
        return workOrderRepository.findAll();
    }
}