package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.SalesHistory;
import com.metawebthree.inventory.infrastructure.persistence.converter.SalesHistoryConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.SalesHistoryDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.SalesHistoryMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public class SalesHistoryRepositoryImpl implements SalesHistoryRepository {

    private final SalesHistoryMapper salesHistoryMapper;
    private final SalesHistoryConverter salesHistoryConverter;

    public SalesHistoryRepositoryImpl(SalesHistoryMapper salesHistoryMapper,
                                      SalesHistoryConverter salesHistoryConverter) {
        this.salesHistoryMapper = salesHistoryMapper;
        this.salesHistoryConverter = salesHistoryConverter;
    }

    @Override
    public Optional<SalesHistory> findById(Long id) {
        SalesHistoryDO dto = salesHistoryMapper.selectById(id);
        return Optional.ofNullable(salesHistoryConverter.toEntity(dto));
    }

    @Override
    public List<SalesHistory> findBySkuAndWarehouse(String skuCode, Long warehouseId) {
        List<SalesHistoryDO> dtos = salesHistoryMapper.selectList(new LambdaQueryWrapper<SalesHistoryDO>()
                .eq(SalesHistoryDO::getSkuCode, skuCode)
                .eq(warehouseId != null, SalesHistoryDO::getWarehouseId, warehouseId));
        return salesHistoryConverter.toEntityList(dtos);
    }

    @Override
    public List<SalesHistory> findBySkuAndWarehouseAndDateRange(
            String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate) {
        List<SalesHistoryDO> dtos = salesHistoryMapper.selectList(new LambdaQueryWrapper<SalesHistoryDO>()
                .eq(SalesHistoryDO::getSkuCode, skuCode)
                .eq(warehouseId != null, SalesHistoryDO::getWarehouseId, warehouseId)
                .ge(startDate != null, SalesHistoryDO::getSalesDate, startDate)
                .le(endDate != null, SalesHistoryDO::getSalesDate, endDate));
        return salesHistoryConverter.toEntityList(dtos);
    }

    @Override
    public SalesHistory save(SalesHistory salesHistory) {
        SalesHistoryDO dto = salesHistoryConverter.toDto(salesHistory);
        if (dto.getId() == null) {
            salesHistoryMapper.insert(dto);
        } else {
            salesHistoryMapper.updateById(dto);
        }
        salesHistory.setId(dto.getId());
        return salesHistory;
    }

    @Override
    public void delete(SalesHistory salesHistory) {
        if (salesHistory.getId() != null) {
            salesHistoryMapper.deleteById(salesHistory.getId());
        }
    }
}