package com.metawebthree.recommendation.domain.repository;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import java.util.List;
import java.util.Optional;

public interface RecommendationRepository {
    Optional<Recommendation> findById(Long id);
    List<Recommendation> findByUserId(Long userId);
    List<Recommendation> findByUserIdAndScene(Long userId, String scene);
    List<Recommendation> findByStatus(Recommendation.RecommendationStatus status);
    Recommendation save(Recommendation recommendation);
    void update(Recommendation recommendation);
    void deleteById(Long id);
    void recordBehavior(Long userId, String skuCode, String behaviorType);
}