package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.metawebthree.warehouse.domain.entity.Warehouse;
import java.util.Optional;

public interface WarehouseRepository {

    Optional<Warehouse> findById(Long id);

    Optional<Warehouse> findByWarehouseCode(String warehouseCode);

    void insert(Warehouse warehouse);

    void update(Warehouse warehouse);

    void delete(Warehouse warehouse);
}