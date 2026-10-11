package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class MesQueryService {

    private final WorkOrderRepository workOrderRepository;
    private final ProductionTaskRepository taskRepository;
    private final ProcessRouteRepository routeRepository;
    private final EquipmentRepository equipmentRepository;

    public MesQueryService(
            WorkOrderRepository workOrderRepository,
            ProductionTaskRepository taskRepository,
            ProcessRouteRepository routeRepository,
            EquipmentRepository equipmentRepository) {
        this.workOrderRepository = workOrderRepository;
        this.taskRepository = taskRepository;
        this.routeRepository = routeRepository;
        this.equipmentRepository = equipmentRepository;
    }

    public Optional<WorkOrder> getWorkOrderById(Long id) {
        return workOrderRepository.findById(id);
    }

    public Optional<WorkOrder> getWorkOrderByNo(String workOrderNo) {
        return workOrderRepository.findByWorkOrderNo(workOrderNo);
    }

    public List<WorkOrder> getWorkOrdersByStatus(WorkOrder.WorkOrderStatus status) {
        return workOrderRepository.findByStatus(status);
    }

    public List<WorkOrder> getWorkshopWorkOrders(String workshopId) {
        return workOrderRepository.findByWorkshopId(workshopId);
    }

    public Optional<ProductionTask> getTaskById(Long id) {
        return taskRepository.findById(id);
    }

    public List<ProductionTask> getTasksByWorkOrder(Long workOrderId) {
        return taskRepository.findByWorkOrderId(workOrderId);
    }

    public Optional<ProcessRoute> getRouteById(Long id) {
        return routeRepository.findById(id);
    }

    public List<ProcessRoute> getRoutesByProduct(String productCode) {
        return routeRepository.findByProductCode(productCode);
    }

    public Optional<Equipment> getEquipmentById(Long id) {
        return equipmentRepository.findById(id);
    }

    public List<Equipment> getWorkshopEquipment(String workshopId) {
        return equipmentRepository.findByWorkshopId(workshopId);
    }

    public List<Equipment> getEquipmentByStatusCode(String statusCode) {
        return equipmentRepository.findByStatusCode(statusCode);
    }
}