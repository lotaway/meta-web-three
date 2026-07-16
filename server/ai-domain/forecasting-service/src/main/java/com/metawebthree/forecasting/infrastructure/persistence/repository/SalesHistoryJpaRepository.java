package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
public interface SalesHistoryJpaRepository extends JpaRepository<SalesHistory, Long> {

    List<SalesHistory> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId);

    List<SalesHistory> findBySkuCodeAndWarehouseIdAndSalesDateBetweenOrderBySalesDateAsc(
        String skuCode, Long warehouseId, LocalDate startDate, LocalDate endDate);
}
