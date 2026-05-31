package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckRecord;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckRecordRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class StockCheckRecordRepositoryImpl implements StockCheckRecordRepository {

    private final Map<Long, StockCheckRecord> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public StockCheckRecord save(StockCheckRecord record) {
        if (record.getId() == null) {
            record.setId(idGenerator.getAndIncrement());
            record.setVersion(0);
        } else {
            record.setVersion(record.getVersion() + 1);
        }
        storage.put(record.getId(), record);
        return record;
    }

    @Override
    public Optional<StockCheckRecord> findById(Long id) {
        return Optional.ofNullable(storage.get(id))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()));
    }

    @Override
    public List<StockCheckRecord> findByPlanId(Long planId) {
        return storage.values().stream()
                .filter(r -> planId.equals(r.getPlanId()))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckRecord> findByPlanNo(String planNo) {
        return storage.values().stream()
                .filter(r -> planNo.equals(r.getPlanNo()))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckRecord> findByWarehouseId(Long warehouseId) {
        return storage.values().stream()
                .filter(r -> warehouseId.equals(r.getWarehouseId()))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckRecord> findByStatus(String status) {
        return storage.values().stream()
                .filter(r -> status.equals(r.getStatus()))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckRecord> findBySkuCode(String skuCode) {
        return storage.values().stream()
                .filter(r -> skuCode.equals(r.getSkuCode()))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckRecord> findHasDifference(Long planId) {
        return storage.values().stream()
                .filter(r -> planId.equals(r.getPlanId()))
                .filter(StockCheckRecord::hasDifference)
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}