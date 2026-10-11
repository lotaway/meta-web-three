package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;
import com.metawebthree.digitaltwin.domain.repository.WarehouseRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.WarehouseConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WarehouseDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.WarehouseMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

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
    public List<Warehouse> findAll() {
        return warehouseMapper.selectList(null).stream()
                .map(warehouseConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Warehouse> findByStatus(WarehouseStatus status) {
        LambdaQueryWrapper<WarehouseDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WarehouseDO::getStatus, status.name());
        return warehouseMapper.selectList(wrapper).stream()
                .map(warehouseConverter::toEntity)
                .collect(Collectors.toList());
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

    @Override
    public boolean existsByWarehouseCode(String warehouseCode) {
        LambdaQueryWrapper<WarehouseDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WarehouseDO::getWarehouseCode, warehouseCode);
        return warehouseMapper.selectCount(wrapper) > 0;
    }
}