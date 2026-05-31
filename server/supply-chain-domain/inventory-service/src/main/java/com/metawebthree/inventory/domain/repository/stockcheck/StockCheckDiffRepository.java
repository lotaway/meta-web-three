package com.metawebthree.inventory.domain.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckDiff;

import java.util.List;
import java.util.Optional;

public interface StockCheckDiffRepository {

    StockCheckDiff save(StockCheckDiff diff);

    Optional<StockCheckDiff> findById(Long id);

    List<StockCheckDiff> findByPlanId(Long planId);

    List<StockCheckDiff> findByPlanNo(String planNo);

    List<StockCheckDiff> findByWarehouseId(Long warehouseId);

    List<StockCheckDiff> findByProcessingStatus(String status);

    List<StockCheckDiff> findByApprovalStatus(String status);

    List<StockCheckDiff> findPendingApproval();

    List<StockCheckDiff> findBySkuCode(String skuCode);

    void deleteById(Long id);
}