package com.metawebthree.production.domain.service;

import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.ProductionSchedule;
import com.metawebthree.production.domain.entity.WorkStation;
import java.util.List;

public interface ProductionDomainService {
    ProductionOrder createOrder(String productCode, String productName, Integer quantity,
                                  ProductionOrder.Priority priority, String workshopCode);
    ProductionOrder scheduleOrder(Long orderId, String productionLineCode);
    ProductionOrder startProduction(Long orderId);
    ProductionOrder pauseProduction(Long orderId);
    ProductionOrder resumeProduction(Long orderId);
    ProductionOrder completeProduction(Long orderId);
    ProductionOrder cancelOrder(Long orderId);
    ProductionSchedule createSchedule(String orderCode, String stationCode, Integer quantity);
    ProductionSchedule startSchedule(Long scheduleId);
    ProductionSchedule completeSchedule(Long scheduleId);
    WorkStation createWorkStation(String stationCode, String stationName, String stationType,
                                   String workshopCode, Integer capacity);
    WorkStation assignOrderToStation(String stationCode, String orderCode);
    WorkStation completeStationOrder(String stationCode);
    List<ProductionOrder> getOrdersByStatus(ProductionOrder.OrderStatus status);
    List<WorkStation> getAvailableStations();
    List<ProductionSchedule> getSchedulesForOrder(String orderCode);
}