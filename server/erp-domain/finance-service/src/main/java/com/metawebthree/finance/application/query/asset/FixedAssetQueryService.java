package com.metawebthree.finance.application.query.asset;

import com.metawebthree.finance.domain.repository.asset.FixedAssetRepository;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDepreciationDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDisposalDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetInventoryDO;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class FixedAssetQueryService {
    private final FixedAssetRepository repository;

    public FixedAssetQueryService(FixedAssetRepository repository) {
        this.repository = repository;
    }

    public FixedAssetDO getAssetById(Long id) {
        return repository.findById(id);
    }

    public FixedAssetDO getAssetByCode(String code) {
        return repository.findByCode(code);
    }

    public List<FixedAssetDO> listAllAssets() {
        return repository.findAll();
    }

    public List<FixedAssetDO> listAssetsByDepartment(Long departmentId) {
        return repository.findByDepartment(departmentId);
    }

    public List<FixedAssetDO> listAssetsByStatus(String status) {
        return repository.findByStatus(status);
    }

    public List<FixedAssetDO> listAssetsByCategory(String category) {
        return repository.findByCategory(category);
    }

    public List<FixedAssetDepreciationDO> listDepreciationByAssetId(Long assetId) {
        return repository.findDepreciationByAssetId(assetId);
    }

    public List<FixedAssetDepreciationDO> listDepreciationByPeriod(String period) {
        return repository.findDepreciationByPeriod(period);
    }

    public List<FixedAssetInventoryDO> listInventoryByStatus(String status) {
        return repository.findInventoryByStatus(status);
    }

    public List<FixedAssetDisposalDO> listDisposalByStatus(String status) {
        return repository.findDisposalByStatus(status);
    }

    public Map<String, Object> getAssetStatistics() {
        List<FixedAssetDO> allAssets = repository.findAll();
        
        BigDecimal totalOriginalValue = BigDecimal.ZERO;
        BigDecimal totalNetValue = BigDecimal.ZERO;
        BigDecimal totalAccumulatedDepreciation = BigDecimal.ZERO;
        
        int activeCount = 0;
        int disposedCount = 0;
        int inUseCount = 0;
        int idleCount = 0;
        
        for (FixedAssetDO asset : allAssets) {
            totalOriginalValue = totalOriginalValue.add(asset.getOriginalValue() != null ? asset.getOriginalValue() : BigDecimal.ZERO);
            totalNetValue = totalNetValue.add(asset.getNetValue() != null ? asset.getNetValue() : BigDecimal.ZERO);
            totalAccumulatedDepreciation = totalAccumulatedDepreciation.add(asset.getAccumulatedDepreciation() != null ? asset.getAccumulatedDepreciation() : BigDecimal.ZERO);
            
            if ("ACTIVE".equals(asset.getStatus())) {
                activeCount++;
            } else if ("DISPOSED".equals(asset.getStatus())) {
                disposedCount++;
            }
            
            if ("IN_USE".equals(asset.getUsageStatus())) {
                inUseCount++;
            } else if ("IDLE".equals(asset.getUsageStatus())) {
                idleCount++;
            }
        }
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalAssets", allAssets.size());
        stats.put("activeCount", activeCount);
        stats.put("disposedCount", disposedCount);
        stats.put("inUseCount", inUseCount);
        stats.put("idleCount", idleCount);
        stats.put("totalOriginalValue", totalOriginalValue);
        stats.put("totalNetValue", totalNetValue);
        stats.put("totalAccumulatedDepreciation", totalAccumulatedDepreciation);
        
        return stats;
    }

    public Map<String, Object> getDepreciationStatistics(String period) {
        List<FixedAssetDepreciationDO> depreciationList = repository.findDepreciationByPeriod(period);
        
        BigDecimal totalDepreciation = BigDecimal.ZERO;
        Map<String, BigDecimal> byMethod = new HashMap<>();
        
        for (FixedAssetDepreciationDO depreciation : depreciationList) {
            totalDepreciation = totalDepreciation.add(depreciation.getDepreciationAmount() != null ? depreciation.getDepreciationAmount() : BigDecimal.ZERO);
            
            String method = depreciation.getDepreciationMethod();
            BigDecimal methodTotal = byMethod.getOrDefault(method, BigDecimal.ZERO);
            byMethod.put(method, methodTotal.add(depreciation.getDepreciationAmount()));
        }
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalDepreciationRecords", depreciationList.size());
        stats.put("totalDepreciationAmount", totalDepreciation);
        stats.put("byMethod", byMethod);
        
        return stats;
    }

    public Map<String, Object> getInventoryStatistics() {
        List<FixedAssetInventoryDO> allInventory = repository.findInventoryByStatus(null);
        
        int totalCount = allInventory.size();
        int pendingCount = 0;
        int completedCount = 0;
        int discrepancyCount = 0;
        
        for (FixedAssetInventoryDO inventory : allInventory) {
            if ("PENDING".equals(inventory.getStatus())) {
                pendingCount++;
            } else if ("COMPLETED".equals(inventory.getStatus())) {
                completedCount++;
            }
            
            if ("DISCREPANCY".equals(inventory.getInventoryResult())) {
                discrepancyCount++;
            }
        }
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalInventoryRecords", totalCount);
        stats.put("pendingCount", pendingCount);
        stats.put("completedCount", completedCount);
        stats.put("discrepancyCount", discrepancyCount);
        
        return stats;
    }
}