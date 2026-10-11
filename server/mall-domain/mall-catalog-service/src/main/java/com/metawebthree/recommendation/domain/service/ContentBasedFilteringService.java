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
public class ContentBasedFilteringService {

    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductSimilarityRepository productSimilarityRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public ContentBasedFilteringService(
            UserBehaviorRepository userBehaviorRepository,
            ProductSimilarityRepository productSimilarityRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.userBehaviorRepository = userBehaviorRepository;
        this.productSimilarityRepository = productSimilarityRepository;
        this.algorithmProperties = algorithmProperties;
    }

    public List<RecommendationResult> contentBasedFiltering(Long userId, int limit) {
        Map<Long, Double> productScores = buildContentBasedProductScores(userId);
        return productScores.entrySet().stream()
            .map(entry -> RecommendationCalculationUtils.createRecommendationResult(
                userId, entry.getKey(), entry.getValue(),
                RecommendationResult.RecommendationAlgorithm.CONTENT_BASED,
                algorithmProperties.getRecommendationExpiryDays()))
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private Map<Long, Double> buildContentBasedProductScores(Long userId) {
        List<UserBehavior> purchaseHistory = userBehaviorRepository
            .findByUserIdAndBehaviorTypeOrderByTimestampDesc(userId, UserBehavior.BehaviorType.PURCHASE);
        Map<Long, Double> productScores = new HashMap<>();
        for (UserBehavior behavior : purchaseHistory) {
            accumulateContentSimilarities(behavior, productScores);
        }
        return productScores;
    }

    private void accumulateContentSimilarities(UserBehavior behavior, Map<Long, Double> productScores) {
        List<ProductSimilarity> contentSimilar =
            productSimilarityRepository.findSimilarProductsByAlgorithm(
                behavior.getProductId(), ProductSimilarity.SimilarityAlgorithm.CONTENT_BASED);
        for (ProductSimilarity similarity : contentSimilar) {
            Long similarProductId = similarity.getProductId1().equals(behavior.getProductId())
                ? similarity.getProductId2()
                : similarity.getProductId1();
            productScores.merge(similarProductId, similarity.getSimilarityScore(), Double::sum);
        }
    }
}
