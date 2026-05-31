package com.metawebthree.inventory.domain.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckPlan;

import java.util.List;
import java.util.Optional;

public interface StockCheckPlanRepository {

    StockCheckPlan save(StockCheckPlan plan);

    Optional<StockCheckPlan> findById(Long id);

    Optional<StockCheckPlan> findByPlanNo(String planNo);

    List<StockCheckPlan> findAll();

    List<StockCheckPlan> findByWarehouseId(Long warehouseId);

    List<StockCheckPlan> findByStatus(String status);

    List<StockCheckPlan> findByWarehouseIdAndStatus(Long warehouseId, String status);

    void deleteById(Long id);
}