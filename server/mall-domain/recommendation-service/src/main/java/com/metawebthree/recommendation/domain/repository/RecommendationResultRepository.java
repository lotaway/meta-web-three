package com.metawebthree.recommendation.domain.repository;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.data.domain.Pageable;

public interface RecommendationResultRepository {
    RecommendationResult save(RecommendationResult recommendationResult);
    List<RecommendationResult> findByUserIdOrderByScoreDesc(Long userId, LocalDateTime now);
    List<RecommendationResult> findByUserIdAndAlgorithm(Long userId, RecommendationResult.RecommendationAlgorithm algorithm, LocalDateTime now);
    List<RecommendationResult> findByUserIdWithPagination(Long userId, LocalDateTime now, Pageable pageable);
    void deleteByExpiresAtBefore(LocalDateTime timestamp);
    void markAsClicked(Long id);
    void markAsPurchased(Long id);
    Long countClicksByProductId(Long productId);
    Long countPurchasesByProductId(Long productId);
    List<RecommendationResult> findAll();
    void deleteById(Long id);
}
