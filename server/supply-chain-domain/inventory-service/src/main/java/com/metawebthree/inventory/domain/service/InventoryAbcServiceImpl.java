package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.AbcClassification;
import com.metawebthree.inventory.domain.entity.AbcClassification.AbcCategory;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.InventoryRecord;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRecordRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class InventoryAbcServiceImpl implements InventoryAbcService {

    private static final BigDecimal A_RATIO = new BigDecimal("0.70");
    private static final BigDecimal B_RATIO = new BigDecimal("0.90");

    private final InventoryRepository inventoryRepository;
    private final InventoryRecordRepository recordRepository;

    public InventoryAbcServiceImpl(InventoryRepository inventoryRepository,
                                    InventoryRecordRepository recordRepository) {
        this.inventoryRepository = inventoryRepository;
        this.recordRepository = recordRepository;
    }

    @Override
    public List<AbcClassification> classify(Long warehouseId, Integer periodDays) {
        List<Inventory> inventories;
        if (warehouseId != null) {
            inventories = inventoryRepository.findByWarehouse(warehouseId);
        } else {
            inventories = inventoryRepository.findAll();
        }

        LocalDateTime endDate = LocalDateTime.now();
        LocalDateTime startDate = endDate.minusDays(periodDays);

        Map<String, List<InventoryRecord>> recordsBySku = new HashMap<>();
        for (Inventory inv : inventories) {
            List<InventoryRecord> records = recordRepository.findBySkuCodeAndDateRange(
                    inv.getSkuCode(), inv.getWarehouseId(), startDate, endDate);
            recordsBySku.put(inv.getSkuCode(), records);
        }

        List<SkuValue> skuValues = inventories.stream()
                .map(inv -> calculateSkuValue(inv, recordsBySku.get(inv.getSkuCode()), periodDays))
                .filter(sv -> sv.totalValue.compareTo(BigDecimal.ZERO) > 0)
                .sorted(Comparator.comparing(SkuValue::getTotalValue).reversed())
                .collect(Collectors.toList());

        if (skuValues.isEmpty()) {
            return Collections.emptyList();
        }

        BigDecimal totalValue = skuValues.stream()
                .map(SkuValue::getTotalValue)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        BigDecimal cumulativeValue = BigDecimal.ZERO;
        int rank = 1;
        List<AbcClassification> result = new ArrayList<>();

        for (SkuValue sv : skuValues) {
            cumulativeValue = cumulativeValue.add(sv.getTotalValue());
            BigDecimal valueRatio = cumulativeValue.divide(totalValue, 4, RoundingMode.HALF_UP);

            AbcCategory category = classifyByRatio(valueRatio);

            AbcClassification classification = new AbcClassification();
            classification.setSkuCode(sv.getSkuCode());
            classification.setCategory(category);
            classification.setTotalValue(sv.getTotalValue());
            classification.setTurnoverRate(sv.getTurnoverRate());
            classification.setRank(rank++);

            result.add(classification);
        }

        return result;
    }

    private AbcCategory classifyByRatio(BigDecimal valueRatio) {
        if (valueRatio.compareTo(A_RATIO) <= 0) {
            return AbcCategory.A;
        } else if (valueRatio.compareTo(B_RATIO) <= 0) {
            return AbcCategory.B;
        } else {
            return AbcCategory.C;
        }
    }

    private SkuValue calculateSkuValue(Inventory inv, List<InventoryRecord> records, Integer periodDays) {
        BigDecimal unitCost = inv.getUnitCost() != null ? inv.getUnitCost() : BigDecimal.ZERO;
        BigDecimal totalValue = unitCost.multiply(BigDecimal.valueOf(inv.getTotalQuantity()));

        BigDecimal turnoverRate = BigDecimal.ZERO;
        if (records != null && !records.isEmpty()) {
            int consumedQty = records.stream()
                    .filter(r -> r.getQuantity() < 0)
                    .mapToInt(r -> Math.abs(r.getQuantity()))
                    .sum();

            BigDecimal avgInventory = BigDecimal.valueOf(inv.getTotalQuantity() + consumedQty)
                    .divide(BigDecimal.valueOf(2), 4, RoundingMode.HALF_UP);

            if (avgInventory.compareTo(BigDecimal.ZERO) > 0) {
                BigDecimal periodsPerYear = BigDecimal.valueOf(360)
                        .divide(BigDecimal.valueOf(periodDays), 4, RoundingMode.HALF_UP);
                turnoverRate = BigDecimal.valueOf(consumedQty)
                        .divide(avgInventory, 4, RoundingMode.HALF_UP)
                        .multiply(periodsPerYear);
            }
        }

        return new SkuValue(inv.getSkuCode(), totalValue, turnoverRate);
    }

    private static class SkuValue {
        private final String skuCode;
        private final BigDecimal totalValue;
        private final BigDecimal turnoverRate;

        SkuValue(String skuCode, BigDecimal totalValue, BigDecimal turnoverRate) {
            this.skuCode = skuCode;
            this.totalValue = totalValue;
            this.turnoverRate = turnoverRate;
        }

        String getSkuCode() {
            return skuCode;
        }

        BigDecimal getTotalValue() {
            return totalValue;
        }

        BigDecimal getTurnoverRate() {
            return turnoverRate;
        }
    }
}