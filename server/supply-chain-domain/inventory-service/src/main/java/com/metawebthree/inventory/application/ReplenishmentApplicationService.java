package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.ReplenishmentRecommendationDTO;
import java.util.List;

public interface ReplenishmentApplicationService {

    ReplenishmentRecommendationDTO generateRecommendation(String skuCode, Long warehouseId, Integer daysToAnalyze);

    List<ReplenishmentRecommendationDTO> generateRecommendationsForWarehouse(Long warehouseId, Integer daysToAnalyze);

    List<ReplenishmentRecommendationDTO> getPendingRecommendations();

    ReplenishmentRecommendationDTO approveRecommendation(Long recommendationId);

    ReplenishmentRecommendationDTO rejectRecommendation(Long recommendationId);

    ReplenishmentRecommendationDTO queryById(Long id);

    List<ReplenishmentRecommendationDTO> queryByWarehouse(Long warehouseId);
}