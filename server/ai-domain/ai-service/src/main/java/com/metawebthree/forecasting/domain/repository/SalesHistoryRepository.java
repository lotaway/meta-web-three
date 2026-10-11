package com.metawebthree.forecasting.domain.repository;

import com.metawebthree.forecasting.domain.entity.SalesHistory;
import java.time.LocalDate;
import java.util.List;

public interface SalesHistoryRepository {
    List<SalesHistory> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId);
    List<SalesHistory> findBySkuCodeAndWarehouseIdAndSalesDateBetween(
        String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate);
    List<SalesHistory> findRecentBySkuCodeAndWarehouseId(
        String skuCode, Long warehouseId, Integer days);
    void save(SalesHistory salesHistory);
    void saveBatch(List<SalesHistory> salesHistoryList);
}