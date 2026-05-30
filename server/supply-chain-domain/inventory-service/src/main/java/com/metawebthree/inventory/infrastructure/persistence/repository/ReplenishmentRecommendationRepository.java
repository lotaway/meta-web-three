package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import java.util.List;
import java.util.Optional;

public interface ReplenishmentRecommendationRepository {

    Optional<ReplenishmentRecommendation> findById(Long id);

    List<ReplenishmentRecommendation> findByWarehouse(Long warehouseId);

    List<ReplenishmentRecommendation> findByStatus(String status);

    List<ReplenishmentRecommendation> findPendingRecommendations();

    ReplenishmentRecommendation save(ReplenishmentRecommendation recommendation);

    void delete(ReplenishmentRecommendation recommendation);
}