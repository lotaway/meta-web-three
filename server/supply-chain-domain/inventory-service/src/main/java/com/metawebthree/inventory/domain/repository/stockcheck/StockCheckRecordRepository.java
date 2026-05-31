package com.metawebthree.inventory.domain.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckRecord;

import java.util.List;
import java.util.Optional;

public interface StockCheckRecordRepository {

    StockCheckRecord save(StockCheckRecord record);

    Optional<StockCheckRecord> findById(Long id);

    List<StockCheckRecord> findByPlanId(Long planId);

    List<StockCheckRecord> findByPlanNo(String planNo);

    List<StockCheckRecord> findByWarehouseId(Long warehouseId);

    List<StockCheckRecord> findByStatus(String status);

    List<StockCheckRecord> findBySkuCode(String skuCode);

    List<StockCheckRecord> findHasDifference(Long planId);

    void deleteById(Long id);
}