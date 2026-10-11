package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class PopularityBasedRecommendationService {

    private final RecommendationResultRepository recommendationResultRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public PopularityBasedRecommendationService(
            RecommendationResultRepository recommendationResultRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.recommendationResultRepository = recommendationResultRepository;
        this.algorithmProperties = algorithmProperties;
    }

    public List<RecommendationResult> popularityBasedRecommendation(Long userId, int limit) {
        List<RecommendationResult> results = recommendationResultRepository
            .findByUserIdAndAlgorithm(userId,
                RecommendationResult.RecommendationAlgorithm.POPULARITY, LocalDateTime.now());
        return results.stream()
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }
}
