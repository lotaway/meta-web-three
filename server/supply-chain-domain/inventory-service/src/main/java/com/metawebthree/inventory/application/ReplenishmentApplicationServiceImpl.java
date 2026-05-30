package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.ReplenishmentRecommendationDTO;
import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.service.ReplenishmentDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.ReplenishmentRecommendationRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.SalesHistoryRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ReplenishmentApplicationServiceImpl implements ReplenishmentApplicationService {

    private final ReplenishmentDomainService domainService;
    private final InventoryRepository inventoryRepository;
    private final ReplenishmentRecommendationRepository recommendationRepository;
    private final SalesHistoryRepository salesHistoryRepository;

    public ReplenishmentApplicationServiceImpl(
            ReplenishmentDomainService domainService,
            InventoryRepository inventoryRepository,
            ReplenishmentRecommendationRepository recommendationRepository,
            SalesHistoryRepository salesHistoryRepository) {
        this.domainService = domainService;
        this.inventoryRepository = inventoryRepository;
        this.recommendationRepository = recommendationRepository;
        this.salesHistoryRepository = salesHistoryRepository;
    }

    @Override
    public ReplenishmentRecommendationDTO generateRecommendation(
            String skuCode, Long warehouseId, Integer daysToAnalyze) {
        Inventory inventory = inventoryRepository.findBySkuAndWarehouse(skuCode, warehouseId)
                .orElseThrow(() -> new IllegalArgumentException("Inventory not found"));
        Integer avgSales = calculateHistoricalSales(skuCode, warehouseId, daysToAnalyze);
        ReplenishmentRecommendation rec = domainService.generateRecommendation(inventory, avgSales);
        return toDTO(rec);
    }

    @Override
    public List<ReplenishmentRecommendationDTO> generateRecommendationsForWarehouse(
            Long warehouseId, Integer daysToAnalyze) {
        List<ReplenishmentRecommendation> recommendations = 
                domainService.generateRecommendationsForWarehouse(warehouseId, daysToAnalyze);
        return recommendations.stream().map(this::toDTO).collect(Collectors.toList());
    }

    @Override
    public List<ReplenishmentRecommendationDTO> getPendingRecommendations() {
        List<ReplenishmentRecommendation> recommendations = domainService.getPendingRecommendations();
        return recommendations.stream().map(this::toDTO).collect(Collectors.toList());
    }

    @Override
    public ReplenishmentRecommendationDTO approveRecommendation(Long recommendationId) {
        ReplenishmentRecommendation rec = domainService.approveRecommendation(recommendationId);
        return toDTO(rec);
    }

    @Override
    public ReplenishmentRecommendationDTO rejectRecommendation(Long recommendationId) {
        ReplenishmentRecommendation rec = domainService.rejectRecommendation(recommendationId);
        return toDTO(rec);
    }

    @Override
    public ReplenishmentRecommendationDTO queryById(Long id) {
        ReplenishmentRecommendation rec = recommendationRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Replenishment recommendation not found: " + id));
        return toDTO(rec);
    }

    @Override
    public List<ReplenishmentRecommendationDTO> queryByWarehouse(Long warehouseId) {
        return List.of();
    }

    private Integer calculateHistoricalSales(String skuCode, Long warehouseId, Integer daysToAnalyze) {
        if (daysToAnalyze == null || daysToAnalyze <= 0) {
            daysToAnalyze = 30;
        }
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(daysToAnalyze);
        List<com.metawebthree.inventory.domain.entity.SalesHistory> salesHistoryList = 
                salesHistoryRepository.findBySkuAndWarehouseAndDateRange(skuCode, warehouseId, startDate, endDate);
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            return 0;
        }
        int totalQuantity = salesHistoryList.stream()
                .mapToInt(com.metawebthree.inventory.domain.entity.SalesHistory::getQuantity)
                .sum();
        return totalQuantity / daysToAnalyze;
    }

    private ReplenishmentRecommendationDTO toDTO(ReplenishmentRecommendation entity) {
        ReplenishmentRecommendationDTO dto = new ReplenishmentRecommendationDTO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setCurrentStock(entity.getCurrentStock());
        dto.setSafetyStock(entity.getSafetyStock());
        dto.setLeadTimeDays(entity.getLeadTimeDays());
        dto.setAverageDailySales(entity.getAverageDailySales());
        dto.setRecommendedQuantity(entity.getRecommendedQuantity());
        dto.setRecommendationType(entity.getRecommendationType());
        dto.setStatus(entity.getStatus());
        dto.setIsUrgent(entity.isUrgent());
        dto.setGeneratedAt(entity.getGeneratedAt());
        return dto;
    }
}