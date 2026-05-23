package com.metawebthree.production.domain.service;

import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.ProductionSchedule;
import com.metawebthree.production.domain.entity.WorkStation;
import com.metawebthree.production.domain.repository.ProductionOrderRepository;
import com.metawebthree.production.domain.repository.WorkStationRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Service
public class ProductionDomainServiceImpl implements ProductionDomainService {
    private static final Logger logger = LoggerFactory.getLogger(ProductionDomainServiceImpl.class);
    
    private final ProductionOrderRepository orderRepository;
    private final WorkStationRepository stationRepository;

    public ProductionDomainServiceImpl(ProductionOrderRepository orderRepository,
                                        WorkStationRepository stationRepository) {
        this.orderRepository = orderRepository;
        this.stationRepository = stationRepository;
    }

    @Override
    public ProductionOrder createOrder(String productCode, String productName, Integer quantity,
                                         ProductionOrder.Priority priority, String workshopCode) {
        ProductionOrder order = new ProductionOrder();
        order.setOrderCode("PO-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        order.setProductCode(productCode);
        order.setProductName(productName);
        order.setQuantityPlanned(quantity);
        order.setPriority(priority != null ? priority : ProductionOrder.Priority.NORMAL);
        order.setWorkshopCode(workshopCode);
        order.setStatus(ProductionOrder.OrderStatus.PENDING);
        
        logger.info("Created production order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder scheduleOrder(Long orderId, String productionLineCode) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        if (order.getStatus() != ProductionOrder.OrderStatus.PENDING) {
            throw new IllegalStateException("Can only schedule pending orders");
        }
        
        order.setProductionLineCode(productionLineCode);
        order.setStatus(ProductionOrder.OrderStatus.SCHEDULED);
        order.setUpdatedAt(LocalDateTime.now());
        
        logger.info("Scheduled order {} to line {}", order.getOrderCode(), productionLineCode);
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder startProduction(Long orderId) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        order.startProduction();
        
        logger.info("Started production for order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder pauseProduction(Long orderId) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        order.pauseProduction();
        
        logger.info("Paused production for order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder resumeProduction(Long orderId) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        order.resumeProduction();
        
        logger.info("Resumed production for order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder completeProduction(Long orderId) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        order.completeProduction();
        
        logger.info("Completed production for order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionOrder cancelOrder(Long orderId) {
        ProductionOrder order = orderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));
        
        order.cancelOrder();
        
        logger.info("Cancelled order: {}", order.getOrderCode());
        return orderRepository.save(order);
    }

    @Override
    public ProductionSchedule createSchedule(String orderCode, String stationCode, Integer quantity) {
        ProductionSchedule schedule = new ProductionSchedule();
        schedule.setScheduleCode("SCH-" + UUID.randomUUID().toString().substring(0, 6).toUpperCase());
        schedule.setOrderCode(orderCode);
        schedule.setStationCode(stationCode);
        schedule.setPlannedQuantity(quantity);
        schedule.setStatus(ProductionSchedule.ScheduleStatus.PENDING);
        
        logger.info("Created schedule: {} for order {}", schedule.getScheduleCode(), orderCode);
        return schedule;
    }

    @Override
    public ProductionSchedule startSchedule(Long scheduleId) {
        logger.info("Started schedule: {}", scheduleId);
        return null;
    }

    @Override
    public ProductionSchedule completeSchedule(Long scheduleId) {
        logger.info("Completed schedule: {}", scheduleId);
        return null;
    }

    @Override
    public WorkStation createWorkStation(String stationCode, String stationName, String stationType,
                                          String workshopCode, Integer capacity) {
        WorkStation station = new WorkStation();
        station.setStationCode(stationCode);
        station.setStationName(stationName);
        station.setStationType(stationType);
        station.setWorkshopCode(workshopCode);
        station.setCapacity(capacity);
        station.setStatus(WorkStation.StationStatus.IDLE);
        
        logger.info("Created work station: {}", stationCode);
        return stationRepository.save(station);
    }

    @Override
    public WorkStation assignOrderToStation(String stationCode, String orderCode) {
        WorkStation station = stationRepository.findByStationCode(stationCode)
            .orElseThrow(() -> new IllegalArgumentException("Station not found: " + stationCode));
        
        station.assignOrder(orderCode);
        
        logger.info("Assigned order {} to station {}", orderCode, stationCode);
        return stationRepository.save(station);
    }

    @Override
    public WorkStation completeStationOrder(String stationCode) {
        WorkStation station = stationRepository.findByStationCode(stationCode)
            .orElseThrow(() -> new IllegalArgumentException("Station not found: " + stationCode));
        
        station.completeOrder();
        
        logger.info("Completed order at station: {}", stationCode);
        return stationRepository.save(station);
    }

    @Override
    public List<ProductionOrder> getOrdersByStatus(ProductionOrder.OrderStatus status) {
        return orderRepository.findByStatus(status);
    }

    @Override
    public List<WorkStation> getAvailableStations() {
        return stationRepository.findAvailableStations();
    }

    @Override
    public List<ProductionSchedule> getSchedulesForOrder(String orderCode) {
        return List.of();
    }
}