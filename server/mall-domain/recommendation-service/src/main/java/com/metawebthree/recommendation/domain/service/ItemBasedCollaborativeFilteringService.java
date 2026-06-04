package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class ItemBasedCollaborativeFilteringService {

    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductSimilarityRepository productSimilarityRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public ItemBasedCollaborativeFilteringService(
            UserBehaviorRepository userBehaviorRepository,
            ProductSimilarityRepository productSimilarityRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.userBehaviorRepository = userBehaviorRepository;
        this.productSimilarityRepository = productSimilarityRepository;
        this.algorithmProperties = algorithmProperties;
    }

    public List<RecommendationResult> itemBasedCollaborativeFiltering(Long userId, int limit) {
        Map<Long, Double> productScores = buildItemBasedProductScores(userId);
        return productScores.entrySet().stream()
            .map(entry -> RecommendationCalculationUtils.createRecommendationResult(
                userId, entry.getKey(), entry.getValue(),
                RecommendationResult.RecommendationAlgorithm.ITEM_BASED_CF,
                algorithmProperties.getRecommendationExpiryDays()))
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private Map<Long, Double> buildItemBasedProductScores(Long userId) {
        List<UserBehavior> userBehaviors = userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId);
        Map<Long, Double> productScores = new HashMap<>();
        for (UserBehavior behavior : userBehaviors) {
            accumulateItemSimilarities(behavior, productScores);
        }
        return productScores;
    }

    private void accumulateItemSimilarities(UserBehavior behavior, Map<Long, Double> productScores) {
        Long productId = behavior.getProductId();
        double behaviorWeight = RecommendationCalculationUtils.getBehaviorWeight(
                algorithmProperties.getBehavior(), behavior.getBehaviorType());
        List<ProductSimilarity> similarProducts = productSimilarityRepository.findSimilarProducts(productId);
        for (ProductSimilarity similarity : similarProducts) {
            Long similarProductId = similarity.getProductId1().equals(productId)
                ? similarity.getProductId2()
                : similarity.getProductId1();
            double score = similarity.getSimilarityScore() * behaviorWeight;
            productScores.merge(similarProductId, score, Double::sum);
        }
    }
}
