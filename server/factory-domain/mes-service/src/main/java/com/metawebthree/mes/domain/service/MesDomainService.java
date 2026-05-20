package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.entity.Equipment;
import java.util.List;

public interface MesDomainService {
    WorkOrder createWorkOrder(String workOrderNo, String productCode, String productName,
                             Integer quantity, String workshopId, String processRouteId);
    
    void releaseWorkOrder(Long workOrderId);
    void startWorkOrder(Long workOrderId);
    void completeWorkOrder(Long workOrderId);
    void cancelWorkOrder(Long workOrderId);
    void updateWorkOrderProgress(Long workOrderId, Integer quantity);
    
    ProductionTask createTask(String taskNo, Long workOrderId, String workstationId,
                              String processCode, Integer quantity, String operatorId);
    void startTask(Long taskId);
    void completeTask(Long taskId, Integer qualified, Integer defective);
    void passQualityCheck(Long taskId);
    void failQualityCheck(Long taskId);
    
    ProcessRoute createProcessRoute(String routeCode, String routeName, String productCode);
    void activateProcessRoute(Long routeId);
    
    Equipment createEquipment(String equipmentCode, String equipmentName,
                              String equipmentType, String workshopId);
    void startEquipmentTask(Long equipmentId, String taskNo);
    void completeEquipmentTask(Long equipmentId);
    void reportEquipmentBreakdown(Long equipmentId);
    void repairEquipment(Long equipmentId);
    
    List<WorkOrder> getWorkshopWorkOrders(String workshopId);
    List<Equipment> getWorkshopEquipment(String workshopId);
}