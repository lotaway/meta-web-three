package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class UserBasedCollaborativeFilteringService {

    private final UserBehaviorRepository userBehaviorRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public UserBasedCollaborativeFilteringService(
            UserBehaviorRepository userBehaviorRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.userBehaviorRepository = userBehaviorRepository;
        this.algorithmProperties = algorithmProperties;
    }

    public List<RecommendationResult> userBasedCollaborativeFiltering(Long userId, int limit) {
        List<Long> similarUserIds = findSimilarUsers(userId, algorithmProperties.getSimilarUserMaxCount());
        Map<Long, Double> productScores = buildUserBasedProductScores(userId, similarUserIds);
        return productScores.entrySet().stream()
            .map(entry -> RecommendationCalculationUtils.createRecommendationResult(
                userId, entry.getKey(), entry.getValue(),
                RecommendationResult.RecommendationAlgorithm.USER_BASED_CF,
                algorithmProperties.getRecommendationExpiryDays()))
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private Map<Long, Double> buildUserBasedProductScores(Long userId, List<Long> similarUserIds) {
        Map<Long, Double> productScores = new HashMap<>();
        for (Long similarUserId : similarUserIds) {
            double similarity = calculateUserSimilarity(userId, similarUserId);
            List<UserBehavior> behaviors = userBehaviorRepository
                .findByUserIdAndBehaviorTypeOrderByTimestampDesc(similarUserId, UserBehavior.BehaviorType.PURCHASE);
            for (UserBehavior behavior : behaviors) {
                Long productId = behavior.getProductId();
                double score = similarity * behavior.getBehaviorValue();
                productScores.merge(productId, score, Double::sum);
            }
        }
        return productScores;
    }

    public double calculateUserSimilarity(Long userId1, Long userId2) {
        List<UserBehavior> behaviors1 = userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId1);
        List<UserBehavior> behaviors2 = userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId2);
        Map<Long, Double> vector1 = RecommendationCalculationUtils.createBehaviorVector(behaviors1, algorithmProperties.getBehavior());
        Map<Long, Double> vector2 = RecommendationCalculationUtils.createBehaviorVector(behaviors2, algorithmProperties.getBehavior());
        return RecommendationCalculationUtils.calculateCosineSimilarity(vector1, vector2);
    }

    private List<Long> findSimilarUsers(Long userId, int maxUsers) {
        List<Long> allUserIds = userBehaviorRepository.findAll().stream()
            .map(UserBehavior::getUserId)
            .distinct()
            .collect(Collectors.toList());
        return allUserIds.stream()
            .filter(id -> !id.equals(userId))
            .sorted((a, b) -> Double.compare(
                calculateUserSimilarity(userId, b),
                calculateUserSimilarity(userId, a)))
            .limit(maxUsers)
            .collect(Collectors.toList());
    }
}
