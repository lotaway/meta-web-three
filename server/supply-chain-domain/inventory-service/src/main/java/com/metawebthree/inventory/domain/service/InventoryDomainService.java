package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.Inventory;
import java.util.Optional;

public interface InventoryDomainService {

    Optional<Inventory> findBySkuAndWarehouse(String skuCode, Long warehouseId);

    Inventory create(String skuCode, Long warehouseId);

    void reserve(Inventory inventory, Integer quantity, String bizId);

    void confirm(Inventory inventory, Integer quantity);

    void cancel(Inventory inventory, Integer quantity);

    void increase(Inventory inventory, Integer quantity);

    void decrease(Inventory inventory, Integer quantity);
}