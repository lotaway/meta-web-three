package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class HybridRecommendationService {

    private final UserBasedCollaborativeFilteringService userBasedCFService;
    private final ItemBasedCollaborativeFilteringService itemBasedCFService;
    private final ContentBasedFilteringService contentBasedService;
    private final PopularityBasedRecommendationService popularityService;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public HybridRecommendationService(
            UserBasedCollaborativeFilteringService userBasedCFService,
            ItemBasedCollaborativeFilteringService itemBasedCFService,
            ContentBasedFilteringService contentBasedService,
            PopularityBasedRecommendationService popularityService,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.userBasedCFService = userBasedCFService;
        this.itemBasedCFService = itemBasedCFService;
        this.contentBasedService = contentBasedService;
        this.popularityService = popularityService;
        this.algorithmProperties = algorithmProperties;
    }

    public List<RecommendationResult> hybridRecommendation(Long userId, int limit) {
        Map<Long, Double> combinedScores = buildHybridCombinedScores(userId, limit);
        return combinedScores.entrySet().stream()
            .map(entry -> RecommendationCalculationUtils.createRecommendationResult(
                userId, entry.getKey(), entry.getValue(),
                RecommendationResult.RecommendationAlgorithm.HYBRID,
                algorithmProperties.getRecommendationExpiryDays()))
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private Map<Long, Double> buildHybridCombinedScores(Long userId, int limit) {
        List<RecommendationResult> userCfRecs = userBasedCFService.userBasedCollaborativeFiltering(userId, limit * 2);
        List<RecommendationResult> itemCfRecs = itemBasedCFService.itemBasedCollaborativeFiltering(userId, limit * 2);
        List<RecommendationResult> contentRecs = contentBasedService.contentBasedFiltering(userId, limit * 2);
        List<RecommendationResult> popularityRecs = popularityService.popularityBasedRecommendation(userId, limit * 2);
        Map<Long, Double> combinedScores = new HashMap<>();
        RecommendationCalculationUtils.combineRecommendations(combinedScores, userCfRecs, algorithmProperties.getScoring().getCollaborativeWeight());
        RecommendationCalculationUtils.combineRecommendations(combinedScores, itemCfRecs, algorithmProperties.getScoring().getContentWeight());
        RecommendationCalculationUtils.combineRecommendations(combinedScores, contentRecs, algorithmProperties.getScoring().getAiModelWeight());
        RecommendationCalculationUtils.combineRecommendations(combinedScores, popularityRecs, algorithmProperties.getScoring().getPopularityWeight());
        return combinedScores;
    }
}
