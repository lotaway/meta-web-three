package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.SalesHistory;
import com.metawebthree.inventory.infrastructure.persistence.repository.ReplenishmentRecommendationRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.SalesHistoryRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ReplenishmentDomainServiceImpl implements ReplenishmentDomainService {

    private final InventoryRepository inventoryRepository;
    private final ReplenishmentRecommendationRepository recommendationRepository;
    private final SalesHistoryRepository salesHistoryRepository;

    public ReplenishmentDomainServiceImpl(
            InventoryRepository inventoryRepository,
            ReplenishmentRecommendationRepository recommendationRepository,
            SalesHistoryRepository salesHistoryRepository) {
        this.inventoryRepository = inventoryRepository;
        this.recommendationRepository = recommendationRepository;
        this.salesHistoryRepository = salesHistoryRepository;
    }

    @Override
    public ReplenishmentRecommendation generateRecommendation(Inventory inventory, Integer averageDailySales) {
        ReplenishmentRecommendation recommendation = new ReplenishmentRecommendation();
        recommendation.setSkuCode(inventory.getSkuCode());
        recommendation.setWarehouseId(inventory.getWarehouseId());
        recommendation.setCurrentStock(inventory.getAvailableQuantity());
        recommendation.setSafetyStock(inventory.getSafetyStock());
        recommendation.setLeadTimeDays(inventory.getLeadTimeDays());
        recommendation.setAverageDailySales(averageDailySales);
        recommendation.generate();
        return recommendationRepository.save(recommendation);
    }

    @Override
    public List<ReplenishmentRecommendation> generateRecommendationsForWarehouse(Long warehouseId, Integer daysToAnalyze) {
        List<Inventory> inventories = inventoryRepository.findByWarehouse(warehouseId);
        List<ReplenishmentRecommendation> recommendations = new ArrayList<>();
        for (Inventory inventory : inventories) {
            if (inventory.getSafetyStock() != null && inventory.getSafetyStock() > 0) {
                Integer averageSales = calculateAverageSales(inventory.getSkuCode(), warehouseId, daysToAnalyze);
                ReplenishmentRecommendation rec = generateRecommendation(inventory, averageSales);
                recommendations.add(rec);
            }
        }
        return recommendations.stream()
                .filter(r -> r.getRecommendedQuantity() > 0)
                .collect(Collectors.toList());
    }

    @Override
    public List<ReplenishmentRecommendation> getPendingRecommendations() {
        return recommendationRepository.findPendingRecommendations();
    }

    @Override
    public ReplenishmentRecommendation approveRecommendation(Long recommendationId) {
        ReplenishmentRecommendation recommendation = recommendationRepository.findById(recommendationId)
                .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        recommendation.approve();
        return recommendationRepository.save(recommendation);
    }

    @Override
    public ReplenishmentRecommendation rejectRecommendation(Long recommendationId) {
        ReplenishmentRecommendation recommendation = recommendationRepository.findById(recommendationId)
                .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        recommendation.reject();
        return recommendationRepository.save(recommendation);
    }

    private Integer calculateAverageSales(String skuCode, Long warehouseId, Integer daysToAnalyze) {
        if (daysToAnalyze == null || daysToAnalyze <= 0) {
            daysToAnalyze = 30;
        }
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(daysToAnalyze);
        List<SalesHistory> salesHistoryList = 
                salesHistoryRepository.findBySkuAndWarehouseAndDateRange(skuCode, warehouseId, startDate, endDate);
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            return 0;
        }
        int totalQuantity = salesHistoryList.stream()
                .mapToInt(SalesHistory::getQuantity)
                .sum();
        return totalQuantity / daysToAnalyze;
    }
}