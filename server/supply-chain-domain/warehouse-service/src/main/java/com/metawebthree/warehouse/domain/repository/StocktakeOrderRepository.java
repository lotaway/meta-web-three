package com.metawebthree.warehouse.domain.repository;

import com.metawebthree.warehouse.domain.entity.StocktakeOrder;
import java.util.List;
import java.util.Optional;

public interface StocktakeOrderRepository {
    Optional<StocktakeOrder> findById(Long id);
    Optional<StocktakeOrder> findByOrderNo(String orderNo);
    List<StocktakeOrder> findByWarehouseId(Long warehouseId);
    List<StocktakeOrder> findByStatus(String status);
    List<StocktakeOrder> findByWarehouseIdAndStatus(Long warehouseId, String status);
    void insert(StocktakeOrder order);
    void update(StocktakeOrder order);
    void delete(Long id);
}