package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.SalesHistory;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface SalesHistoryRepository {

    Optional<SalesHistory> findById(Long id);

    List<SalesHistory> findBySkuAndWarehouse(String skuCode, Long warehouseId);

    List<SalesHistory> findBySkuAndWarehouseAndDateRange(
            String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate);

    SalesHistory save(SalesHistory salesHistory);

    void delete(SalesHistory salesHistory);
}