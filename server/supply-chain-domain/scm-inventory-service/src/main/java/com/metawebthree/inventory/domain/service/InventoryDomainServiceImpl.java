package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class InventoryDomainServiceImpl implements InventoryDomainService {

    private final InventoryRepository inventoryRepository;

    public InventoryDomainServiceImpl(InventoryRepository inventoryRepository) {
        this.inventoryRepository = inventoryRepository;
    }

    @Override
    public Optional<Inventory> findBySkuAndWarehouse(String skuCode, Long warehouseId) {
        return inventoryRepository.findBySkuAndWarehouse(skuCode, warehouseId);
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
        return inventoryRepository.save(inventory);
    }

    @Override
    public void reserve(Inventory inventory, Integer quantity, String bizId) {
        inventory.reserve(quantity);
        inventoryRepository.save(inventory);
    }

    @Override
    public void confirm(Inventory inventory, Integer quantity) {
        inventory.confirmReserve(quantity);
        inventoryRepository.save(inventory);
    }

    @Override
    public void cancel(Inventory inventory, Integer quantity) {
        inventory.cancelReserve(quantity);
        inventoryRepository.save(inventory);
    }

    @Override
    public void increase(Inventory inventory, Integer quantity) {
        inventory.increase(quantity);
        inventoryRepository.save(inventory);
    }

    @Override
    public void decrease(Inventory inventory, Integer quantity) {
        inventory.decrease(quantity);
        inventoryRepository.save(inventory);
    }
}