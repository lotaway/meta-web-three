package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
public class SalesHistoryRepositoryImpl implements SalesHistoryRepository {

    private final SalesHistoryJpaRepository jpaRepository;

    public SalesHistoryRepositoryImpl(SalesHistoryJpaRepository jpaRepository) {
        this.jpaRepository = jpaRepository;
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId) {
        return jpaRepository.findBySkuCodeAndWarehouseId(skuCode, warehouseId);
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseIdAndSalesDateBetween(
            String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate) {
        return jpaRepository.findBySkuCodeAndWarehouseIdAndSalesDateBetweenOrderBySalesDateAsc(
            skuCode, warehouseId, startDate, endDate);
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
    public SalesHistory save(SalesHistory salesHistory) {
        return jpaRepository.save(salesHistory);
    }

    @Override
    public void saveBatch(List<SalesHistory> salesHistoryList) {
        jpaRepository.saveAll(salesHistoryList);
    }
}
