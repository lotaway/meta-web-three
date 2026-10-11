package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.InventoryRecord;
import com.metawebthree.inventory.infrastructure.persistence.converter.InventoryRecordConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryRecordDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryRecordMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class InventoryRecordRepositoryImpl implements InventoryRecordRepository {

    private final InventoryRecordMapper recordMapper;
    private final InventoryRecordConverter converter;

    public InventoryRecordRepositoryImpl(InventoryRecordMapper recordMapper, InventoryRecordConverter converter) {
        this.recordMapper = recordMapper;
        this.converter = converter;
    }

    @Override
    public List<InventoryRecord> findByWarehouseAndDateRange(Long warehouseId, LocalDateTime startDate, LocalDateTime endDate) {
        LambdaQueryWrapper<InventoryRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryRecordDO::getWarehouseId, warehouseId)
               .between(InventoryRecordDO::getCreatedAt, startDate, endDate);
        return recordMapper.selectList(wrapper).stream()
                .map(converter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryRecord> findBySkuCodeAndDateRange(String skuCode, Long warehouseId, LocalDateTime startDate, LocalDateTime endDate) {
        LambdaQueryWrapper<InventoryRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryRecordDO::getSkuCode, skuCode)
               .eq(InventoryRecordDO::getWarehouseId, warehouseId)
               .between(InventoryRecordDO::getCreatedAt, startDate, endDate);
        return recordMapper.selectList(wrapper).stream()
                .map(converter::toEntity)
                .collect(Collectors.toList());
    }
}