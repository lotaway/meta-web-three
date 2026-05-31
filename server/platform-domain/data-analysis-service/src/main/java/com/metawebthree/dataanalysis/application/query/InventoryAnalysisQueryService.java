package com.metawebthree.dataanalysis.application.query;

import com.metawebthree.dataanalysis.application.dto.*;
import com.metawebthree.dataanalysis.domain.entity.InventoryAnalysisDO;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.InventoryAnalysisMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class InventoryAnalysisQueryService {

    private final InventoryAnalysisMapper inventoryAnalysisMapper;

    public InventoryOverviewDTO getInventoryOverview() {
        List<InventoryAnalysisDO> allRecords = inventoryAnalysisMapper.selectAll();
        List<InventoryAnalysisDO> lowStockRecords = inventoryAnalysisMapper.selectLowStock();
        List<InventoryAnalysisDO> overstockRecords = inventoryAnalysisMapper.selectOverstock();
        
        InventoryOverviewDTO result = new InventoryOverviewDTO();
        result.setTotalProducts((long) allRecords.size());
        result.setTotalStock(allRecords.stream()
            .mapToLong(r -> r.getCurrentStock() != null ? r.getCurrentStock() : 0L)
            .sum());
        result.setLowStockCount((long) lowStockRecords.size());
        result.setOverstockCount((long) overstockRecords.size());
        
        if (!allRecords.isEmpty()) {
            result.setAvgTurnoverRate(allRecords.stream()
                .filter(r -> r.getTurnoverRate() != null)
                .mapToInt(InventoryAnalysisDO::getTurnoverRate)
                .average()
                .orElse(0.0));
        }
        
        result.setAlerts(generateAlerts(lowStockRecords, overstockRecords));
        
        return result;
    }

    public InventoryAnalysisDTO getProductInventory(String productId) {
        InventoryAnalysisDO record = inventoryAnalysisMapper.selectByProductId(productId);
        return record != null ? toDTO(record) : null;
    }

    public List<InventoryAnalysisDTO> getLowStockProducts() {
        List<InventoryAnalysisDO> records = inventoryAnalysisMapper.selectLowStock();
        return records.stream().map(this::toDTO).collect(Collectors.toList());
    }

    public List<InventoryAnalysisDTO> getOverstockProducts() {
        List<InventoryAnalysisDO> records = inventoryAnalysisMapper.selectOverstock();
        return records.stream().map(this::toDTO).collect(Collectors.toList());
    }

    private List<InventoryAlertDTO> generateAlerts(
            List<InventoryAnalysisDO> lowStock,
            List<InventoryAnalysisDO> overstock) {
        
        List<InventoryAlertDTO> alerts = new ArrayList<>();
        
        for (InventoryAnalysisDO record : lowStock) {
            InventoryAlertDTO alert = new InventoryAlertDTO();
            alert.setProductId(record.getProductId());
            alert.setProductName(record.getProductName());
            alert.setAlertType("LOW_STOCK");
            alert.setCurrentStock(record.getCurrentStock());
            alert.setThreshold(record.getSafetyStock());
            alerts.add(alert);
        }
        
        for (InventoryAnalysisDO record : overstock) {
            InventoryAlertDTO alert = new InventoryAlertDTO();
            alert.setProductId(record.getProductId());
            alert.setProductName(record.getProductName());
            alert.setAlertType("OVERSTOCK");
            alert.setCurrentStock(record.getCurrentStock());
            alert.setThreshold(record.getReorderPoint());
            alerts.add(alert);
        }
        
        return alerts;
    }

    private InventoryAnalysisDTO toDTO(InventoryAnalysisDO record) {
        InventoryAnalysisDTO dto = new InventoryAnalysisDTO();
        dto.setProductId(record.getProductId());
        dto.setProductName(record.getProductName());
        dto.setCategory(record.getCategory());
        dto.setCurrentStock(record.getCurrentStock());
        dto.setSafetyStock(record.getSafetyStock());
        dto.setReorderPoint(record.getReorderPoint());
        dto.setTurnoverRate(record.getTurnoverRate());
        dto.setStockDays(record.getStockDays());
        dto.setStockStatus(record.getStockStatus());
        return dto;
    }
}