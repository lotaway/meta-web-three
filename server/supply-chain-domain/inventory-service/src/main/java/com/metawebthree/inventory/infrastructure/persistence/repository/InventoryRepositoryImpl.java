package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.infrastructure.persistence.converter.InventoryConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InventoryRepositoryImpl implements InventoryRepository {

    private final InventoryMapper inventoryMapper;
    private final InventoryConverter inventoryConverter;

    public InventoryRepositoryImpl(InventoryMapper inventoryMapper, InventoryConverter inventoryConverter) {
        this.inventoryMapper = inventoryMapper;
        this.inventoryConverter = inventoryConverter;
    }

    @Override
    public Optional<Inventory> findById(Long id) {
        InventoryDO inventoryDO = inventoryMapper.selectById(id);
        return Optional.ofNullable(inventoryConverter.toEntity(inventoryDO));
    }

    @Override
    public Optional<Inventory> findBySkuAndWarehouse(String skuCode, Long warehouseId) {
        LambdaQueryWrapper<InventoryDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryDO::getSkuCode, skuCode)
               .eq(InventoryDO::getWarehouseId, warehouseId);
        InventoryDO inventoryDO = inventoryMapper.selectOne(wrapper);
        return Optional.ofNullable(inventoryConverter.toEntity(inventoryDO));
    }

    @Override
    public List<Inventory> findByWarehouse(Long warehouseId) {
        LambdaQueryWrapper<InventoryDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryDO::getWarehouseId, warehouseId);
        return inventoryMapper.selectList(wrapper).stream()
                .map(inventoryConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Inventory> findAll() {
        return inventoryMapper.selectList(null).stream()
                .map(inventoryConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public Inventory save(Inventory inventory) {
        InventoryDO inventoryDO = inventoryConverter.toDO(inventory);
        if (inventory.getId() == null) {
            inventoryMapper.insert(inventoryDO);
            inventory.setId(inventoryDO.getId());
        } else {
            inventoryMapper.updateById(inventoryDO);
        }
        return inventory;
    }

    @Override
    public void delete(Inventory inventory) {
        if (inventory.getId() != null) {
            inventoryMapper.deleteById(inventory.getId());
        }
    }
}