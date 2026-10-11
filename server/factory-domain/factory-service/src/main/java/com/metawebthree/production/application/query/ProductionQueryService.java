package com.metawebthree.production.application.query;

import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.WorkStation;
import com.metawebthree.production.domain.repository.ProductionOrderRepository;
import com.metawebthree.production.domain.repository.WorkStationRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProductionQueryService {
    private static final Logger logger = LoggerFactory.getLogger(ProductionQueryService.class);
    
    private final ProductionOrderRepository orderRepository;
    private final WorkStationRepository stationRepository;

    public ProductionQueryService(ProductionOrderRepository orderRepository,
                                   WorkStationRepository stationRepository) {
        this.orderRepository = orderRepository;
        this.stationRepository = stationRepository;
    }

    public ProductionOrder getOrderById(Long id) {
        return orderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + id));
    }

    public ProductionOrder getOrderByCode(String orderCode) {
        return orderRepository.findByOrderCode(orderCode)
            .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderCode));
    }

    public List<ProductionOrder> getAllOrders() {
        return orderRepository.findAll();
    }

    public List<ProductionOrder> getOrdersByStatus(ProductionOrder.OrderStatus status) {
        return orderRepository.findByStatus(status);
    }

    public List<ProductionOrder> getOrdersByWorkshop(String workshopCode) {
        return orderRepository.findByWorkshopCode(workshopCode);
    }

    public List<ProductionOrder> getOrdersByPriority(ProductionOrder.Priority priority) {
        return orderRepository.findByPriority(priority);
    }

    public WorkStation getStationById(Long id) {
        return stationRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Station not found: " + id));
    }

    public WorkStation getStationByCode(String stationCode) {
        return stationRepository.findByStationCode(stationCode)
            .orElseThrow(() -> new IllegalArgumentException("Station not found: " + stationCode));
    }

    public List<WorkStation> getAllStations() {
        return stationRepository.findAll();
    }

    public List<WorkStation> getStationsByStatus(WorkStation.StationStatus status) {
        return stationRepository.findByStatus(status);
    }

    public List<WorkStation> getAvailableStations() {
        return stationRepository.findAvailableStations();
    }
}