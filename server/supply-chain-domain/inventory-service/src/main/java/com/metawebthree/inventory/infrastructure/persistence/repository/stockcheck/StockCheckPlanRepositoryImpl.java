package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckPlan;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckPlanRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class StockCheckPlanRepositoryImpl implements StockCheckPlanRepository {

    private final Map<Long, StockCheckPlan> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public StockCheckPlan save(StockCheckPlan plan) {
        if (plan.getId() == null) {
            plan.setId(idGenerator.getAndIncrement());
            plan.setVersion(0);
        } else {
            plan.setVersion(plan.getVersion() + 1);
        }
        storage.put(plan.getId(), plan);
        return plan;
    }

    @Override
    public Optional<StockCheckPlan> findById(Long id) {
        return Optional.ofNullable(storage.get(id))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()));
    }

    @Override
    public Optional<StockCheckPlan> findByPlanNo(String planNo) {
        return storage.values().stream()
                .filter(p -> planNo.equals(p.getPlanNo()))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()))
                .findFirst();
    }

    @Override
    public List<StockCheckPlan> findAll() {
        return storage.values().stream()
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckPlan> findByWarehouseId(Long warehouseId) {
        return storage.values().stream()
                .filter(p -> warehouseId.equals(p.getWarehouseId()))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckPlan> findByStatus(String status) {
        return storage.values().stream()
                .filter(p -> status.equals(p.getStatus()))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckPlan> findByWarehouseIdAndStatus(Long warehouseId, String status) {
        return storage.values().stream()
                .filter(p -> warehouseId.equals(p.getWarehouseId()))
                .filter(p -> status.equals(p.getStatus()))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}