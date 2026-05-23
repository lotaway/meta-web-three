package com.metawebthree.production.infrastructure.persistence.repository;

import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.repository.ProductionOrderRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class ProductionOrderRepositoryImpl implements ProductionOrderRepository {
    private final Map<Long, ProductionOrder> storage = new ConcurrentHashMap<>();
    private final Map<String, ProductionOrder> codeIndex = new ConcurrentHashMap<>();
    private Long idGenerator = 1L;

    @Override
    public Optional<ProductionOrder> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<ProductionOrder> findByOrderCode(String orderCode) {
        return Optional.ofNullable(codeIndex.get(orderCode));
    }

    @Override
    public List<ProductionOrder> findByStatus(ProductionOrder.OrderStatus status) {
        return storage.values().stream()
            .filter(o -> o.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<ProductionOrder> findByWorkshopCode(String workshopCode) {
        return storage.values().stream()
            .filter(o -> workshopCode.equals(o.getWorkshopCode()))
            .collect(Collectors.toList());
    }

    @Override
    public List<ProductionOrder> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public ProductionOrder save(ProductionOrder order) {
        if (order.getId() == null) {
            order.setId(idGenerator++);
        }
        storage.put(order.getId(), order);
        if (order.getOrderCode() != null) {
            codeIndex.put(order.getOrderCode(), order);
        }
        return order;
    }

    @Override
    public void delete(ProductionOrder order) {
        if (order.getId() != null) {
            storage.remove(order.getId());
        }
        if (order.getOrderCode() != null) {
            codeIndex.remove(order.getOrderCode());
        }
    }

    @Override
    public List<ProductionOrder> findByPriority(ProductionOrder.Priority priority) {
        return storage.values().stream()
            .filter(o -> o.getPriority() == priority)
            .collect(Collectors.toList());
    }
}