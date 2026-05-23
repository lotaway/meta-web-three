package com.metawebthree.production.domain.repository;

import com.metawebthree.production.domain.entity.ProductionOrder;
import java.util.List;
import java.util.Optional;

public interface ProductionOrderRepository {
    Optional<ProductionOrder> findById(Long id);
    Optional<ProductionOrder> findByOrderCode(String orderCode);
    List<ProductionOrder> findByStatus(ProductionOrder.OrderStatus status);
    List<ProductionOrder> findByWorkshopCode(String workshopCode);
    List<ProductionOrder> findAll();
    ProductionOrder save(ProductionOrder order);
    void delete(ProductionOrder order);
    List<ProductionOrder> findByPriority(ProductionOrder.Priority priority);
}