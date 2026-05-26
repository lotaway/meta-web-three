package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.infrastructure.persistence.converter.WarehouseConverter;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.WarehouseDO;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.WarehouseMapper;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public class WarehouseRepositoryImpl implements WarehouseRepository {

    private final WarehouseMapper warehouseMapper;
    private final WarehouseConverter warehouseConverter;

    public WarehouseRepositoryImpl(WarehouseMapper warehouseMapper, WarehouseConverter warehouseConverter) {
        this.warehouseMapper = warehouseMapper;
        this.warehouseConverter = warehouseConverter;
    }

    @Override
    public Optional<Warehouse> findById(Long id) {
        WarehouseDO warehouseDO = warehouseMapper.selectById(id);
        return Optional.ofNullable(warehouseConverter.toEntity(warehouseDO));
    }

    @Override
    public Optional<Warehouse> findByWarehouseCode(String warehouseCode) {
        LambdaQueryWrapper<WarehouseDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WarehouseDO::getWarehouseCode, warehouseCode);
        WarehouseDO warehouseDO = warehouseMapper.selectOne(wrapper);
        return Optional.ofNullable(warehouseConverter.toEntity(warehouseDO));
    }

    @Override
    public void insert(Warehouse warehouse) {
        WarehouseDO warehouseDO = warehouseConverter.toDO(warehouse);
        warehouseMapper.insert(warehouseDO);
        warehouse.setId(warehouseDO.getId());
    }

    @Override
    public void update(Warehouse warehouse) {
        WarehouseDO warehouseDO = warehouseConverter.toDO(warehouse);
        warehouseMapper.updateById(warehouseDO);
    }

    @Override
    public void delete(Warehouse warehouse) {
        if (warehouse.getId() != null) {
            warehouseMapper.deleteById(warehouse.getId());
        }
    }
}