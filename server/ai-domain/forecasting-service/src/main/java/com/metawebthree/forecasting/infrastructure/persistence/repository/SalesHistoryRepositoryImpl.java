package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class SalesHistoryRepositoryImpl implements SalesHistoryRepository {

    private final Map<String, List<SalesHistory>> storage = new ConcurrentHashMap<>();

    private String buildKey(String skuCode, Long warehouseId) {
        return skuCode + "_" + warehouseId;
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId) {
        String key = buildKey(skuCode, warehouseId);
        return new ArrayList<>(storage.getOrDefault(key, new ArrayList<>()));
    }

    @Override
    public List<SalesHistory> findBySkuCodeAndWarehouseIdAndSalesDateBetween(
            String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate) {
        return findBySkuCodeAndWarehouseId(skuCode, warehouseId).stream()
            .filter(h -> !h.getSalesDate().isBefore(startDate) && !h.getSalesDate().isAfter(endDate))
            .sorted((a, b) -> a.getSalesDate().compareTo(b.getSalesDate()))
            .collect(Collectors.toList());
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
        String key = buildKey(salesHistory.getSkuCode(), salesHistory.getWarehouseId());
        storage.computeIfAbsent(key, k -> new ArrayList<>()).add(salesHistory);
        return salesHistory;
    }

    @Override
    public void saveBatch(List<SalesHistory> salesHistoryList) {
        for (SalesHistory salesHistory : salesHistoryList) {
            save(salesHistory);
        }
    }
}