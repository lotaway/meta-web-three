package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import com.metawebthree.inventory.domain.entity.Inventory;
import java.util.List;

public interface ReplenishmentDomainService {

    ReplenishmentRecommendation generateRecommendation(Inventory inventory, Integer averageDailySales);

    List<ReplenishmentRecommendation> generateRecommendationsForWarehouse(Long warehouseId, Integer daysToAnalyze);

    List<ReplenishmentRecommendation> getPendingRecommendations();

    ReplenishmentRecommendation approveRecommendation(Long recommendationId);

    ReplenishmentRecommendation rejectRecommendation(Long recommendationId);
}