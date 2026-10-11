package com.metawebthree.production.application.command;

import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.ProductionSchedule;
import com.metawebthree.production.domain.entity.WorkStation;
import com.metawebthree.production.domain.service.ProductionDomainService;
import com.metawebthree.production.infrastructure.event.ProductionEventPublisher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class ProductionCommandService {
    private static final Logger logger = LoggerFactory.getLogger(ProductionCommandService.class);
    
    private final ProductionDomainService domainService;
    private final ProductionEventPublisher eventPublisher;

    public ProductionCommandService(ProductionDomainService domainService,
                                     ProductionEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public ProductionOrder createOrder(String productCode, String productName, Integer quantity,
                                         ProductionOrder.Priority priority, String workshopCode) {
        ProductionOrder order = domainService.createOrder(productCode, productName, quantity, priority, workshopCode);
        eventPublisher.publishOrderCreated(order);
        logger.info("Created order: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder scheduleOrder(Long orderId, String productionLineCode) {
        ProductionOrder order = domainService.scheduleOrder(orderId, productionLineCode);
        eventPublisher.publishOrderScheduled(order);
        logger.info("Scheduled order: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder startProduction(Long orderId) {
        ProductionOrder order = domainService.startProduction(orderId);
        eventPublisher.publishProductionStarted(order);
        logger.info("Started production: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder pauseProduction(Long orderId) {
        ProductionOrder order = domainService.pauseProduction(orderId);
        eventPublisher.publishProductionPaused(order);
        logger.info("Paused production: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder resumeProduction(Long orderId) {
        ProductionOrder order = domainService.resumeProduction(orderId);
        eventPublisher.publishProductionResumed(order);
        logger.info("Resumed production: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder completeProduction(Long orderId) {
        ProductionOrder order = domainService.completeProduction(orderId);
        eventPublisher.publishProductionCompleted(order);
        logger.info("Completed production: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionOrder cancelOrder(Long orderId) {
        ProductionOrder order = domainService.cancelOrder(orderId);
        eventPublisher.publishOrderCancelled(order);
        logger.info("Cancelled order: {}", order.getOrderCode());
        return order;
    }

    @Transactional
    public ProductionSchedule createSchedule(String orderCode, String stationCode, Integer quantity) {
        ProductionSchedule schedule = domainService.createSchedule(orderCode, stationCode, quantity);
        logger.info("Created schedule: {}", schedule.getScheduleCode());
        return schedule;
    }

    @Transactional
    public WorkStation createWorkStation(String stationCode, String stationName, String stationType,
                                          String workshopCode, Integer capacity) {
        WorkStation station = domainService.createWorkStation(stationCode, stationName, stationType, 
            workshopCode, capacity);
        eventPublisher.publishWorkStationCreated(station);
        logger.info("Created work station: {}", stationCode);
        return station;
    }

    @Transactional
    public WorkStation assignOrderToStation(String stationCode, String orderCode) {
        WorkStation station = domainService.assignOrderToStation(stationCode, orderCode);
        eventPublisher.publishOrderAssignedToStation(station, orderCode);
        logger.info("Assigned order {} to station {}", orderCode, stationCode);
        return station;
    }
}