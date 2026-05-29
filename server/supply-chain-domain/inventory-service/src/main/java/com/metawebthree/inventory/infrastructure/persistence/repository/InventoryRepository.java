package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.Inventory;
import java.util.Optional;

public interface InventoryRepository {

    Optional<Inventory> findById(Long id);

    Optional<Inventory> findBySkuAndWarehouse(String skuCode, Long warehouseId);

    Inventory save(Inventory inventory);

    void delete(Inventory inventory);
}