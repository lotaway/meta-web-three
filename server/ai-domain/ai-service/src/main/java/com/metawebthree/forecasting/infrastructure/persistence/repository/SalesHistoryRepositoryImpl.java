package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import com.metawebthree.forecasting.infrastructure.persistence.mapper.SalesHistoryMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
public class SalesHistoryRepositoryImpl implements SalesHistoryRepository {

    private final SalesHistoryMapper mapper;

    public SalesHistoryRepositoryImpl(SalesHistoryMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId) {
        return mapper.selectList(new QueryWrapper<SalesHistory>()
            .eq("sku_code", skuCode).eq("warehouse_id", warehouseId));
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseIdAndSalesDateBetween(
            String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate) {
        return mapper.selectList(new QueryWrapper<SalesHistory>()
            .eq("sku_code", skuCode)
            .eq("warehouse_id", warehouseId)
            .between("sales_date", startDate, endDate)
            .orderByAsc("sales_date"));
    }

    @Override
    public List<SalesHistory> findRecentBySkuCodeAndWarehouseId(
            String skuCode, Long warehouseId, Integer days) {
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days);
        return findBySkuCodeAndWarehouseIdAndSalesDateBetween(
            skuCode, warehouseId, startDate, endDate);
    }

    @Override
    public void save(SalesHistory salesHistory) {
        mapper.insert(salesHistory);
    }

    @Override
    public void saveBatch(List<SalesHistory> salesHistoryList) {
        for (SalesHistory h : salesHistoryList) {
            mapper.insert(h);
        }
    }
}
