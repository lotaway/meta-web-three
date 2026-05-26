package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;

import java.util.List;
import java.util.Optional;

public interface WarehouseRepository {
    Optional<Warehouse> findById(Long id);
    Optional<Warehouse> findByWarehouseCode(String warehouseCode);
    List<Warehouse> findAll();
    List<Warehouse> findByStatus(WarehouseStatus status);
    void insert(Warehouse warehouse);
    void update(Warehouse warehouse);
    void delete(Warehouse warehouse);
    boolean existsByWarehouseCode(String warehouseCode);
}