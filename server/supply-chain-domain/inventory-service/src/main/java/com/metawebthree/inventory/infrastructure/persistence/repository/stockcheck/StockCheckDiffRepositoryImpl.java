package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckDiff;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckDiffRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class StockCheckDiffRepositoryImpl implements StockCheckDiffRepository {

    private final Map<Long, StockCheckDiff> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public StockCheckDiff save(StockCheckDiff diff) {
        if (diff.getId() == null) {
            diff.setId(idGenerator.getAndIncrement());
            diff.setVersion(0);
        } else {
            diff.setVersion(diff.getVersion() + 1);
        }
        storage.put(diff.getId(), diff);
        return diff;
    }

    @Override
    public Optional<StockCheckDiff> findById(Long id) {
        return Optional.ofNullable(storage.get(id))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()));
    }

    @Override
    public List<StockCheckDiff> findByPlanId(Long planId) {
        return storage.values().stream()
                .filter(d -> planId.equals(d.getPlanId()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findByPlanNo(String planNo) {
        return storage.values().stream()
                .filter(d -> planNo.equals(d.getPlanNo()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findByWarehouseId(Long warehouseId) {
        return storage.values().stream()
                .filter(d -> warehouseId.equals(d.getWarehouseId()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findByProcessingStatus(String status) {
        return storage.values().stream()
                .filter(d -> status.equals(d.getProcessingStatus()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findByApprovalStatus(String status) {
        return storage.values().stream()
                .filter(d -> status.equals(d.getApprovalStatus()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findPendingApproval() {
        return storage.values().stream()
                .filter(d -> StockCheckDiff.APPROVAL_STATUS_PENDING.equals(d.getApprovalStatus()))
                .filter(StockCheckDiff::needsApproval)
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiff> findBySkuCode(String skuCode) {
        return storage.values().stream()
                .filter(d -> skuCode.equals(d.getSkuCode()))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}