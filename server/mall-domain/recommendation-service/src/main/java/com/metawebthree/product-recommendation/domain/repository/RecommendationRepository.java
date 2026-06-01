package com.metawebthree.product_recommendation.domain.repository;

import com.metawebthree.product_recommendation.domain.model.Recommendation;
import java.util.List;

public interface RecommendationRepository {
    void save(Recommendation recommendation);
    void batchSave(List<Recommendation> recommendations);
    List<Recommendation> findByUserId(Long userId);
    List<Recommendation> findByUserIdAndType(Long userId, String type);
    void deleteByUserId(Long userId);
    void deleteExpired();
}