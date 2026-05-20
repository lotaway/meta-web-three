package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.Inventory;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class InventoryDomainServiceImpl implements InventoryDomainService {

    @Override
    public Optional<Inventory> findBySkuAndWarehouse(String skuCode, Long warehouseId) {
        return Optional.empty();
    }

    @Override
    public Inventory create(String skuCode, Long warehouseId) {
        Inventory inventory = new Inventory();
        inventory.setSkuCode(skuCode);
        inventory.setWarehouseId(warehouseId);
        inventory.setTotalQuantity(0);
        inventory.setAvailableQuantity(0);
        inventory.setReservedQuantity(0);
        inventory.setDefectiveQuantity(0);
        return inventory;
    }

    @Override
    public void reserve(Inventory inventory, Integer quantity, String bizId) {
        inventory.reserve(quantity);
    }

    @Override
    public void confirm(Inventory inventory, Integer quantity) {
        inventory.confirmReserve(quantity);
    }

    @Override
    public void cancel(Inventory inventory, Integer quantity) {
        inventory.cancelReserve(quantity);
    }

    @Override
    public void increase(Inventory inventory, Integer quantity) {
        inventory.increase(quantity);
    }

    @Override
    public void decrease(Inventory inventory, Integer quantity) {
        inventory.decrease(quantity);
    }
}