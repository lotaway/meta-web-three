package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import java.time.LocalDateTime;
import java.util.*;

public final class RecommendationCalculationUtils {

    private static final String RECOMMENDED_BASED_ON = "Recommended based on ";

    private RecommendationCalculationUtils() {}

    public static double getBehaviorWeight(
            RecommendationAlgorithmProperties.Behavior behavior,
            UserBehavior.BehaviorType behaviorType) {
        return switch (behaviorType) {
            case PURCHASE -> behavior.getPurchaseWeight();
            case CART -> behavior.getCartWeight();
            case COLLECT -> behavior.getCollectWeight();
            case CLICK -> behavior.getClickWeight();
            case VIEW -> behavior.getViewWeight();
            default -> behavior.getDefaultWeight();
        };
    }

    public static RecommendationResult createRecommendationResult(
            Long userId, Long productId, Double score,
            RecommendationResult.RecommendationAlgorithm algorithm,
            int expiryDays) {
        RecommendationResult result = new RecommendationResult();
        result.setUserId(userId);
        result.setProductId(productId);
        result.setScore(score);
        result.setAlgorithm(algorithm);
        result.setReason(RECOMMENDED_BASED_ON + algorithm.name());
        result.setCreatedAt(LocalDateTime.now());
        result.setExpiresAt(LocalDateTime.now().plusDays(expiryDays));
        result.setIsClicked(false);
        result.setIsPurchased(false);
        return result;
    }

    public static void combineRecommendations(
            Map<Long, Double> combinedScores,
            List<RecommendationResult> recommendations,
            double weight) {
        for (int i = 0; i < recommendations.size(); i++) {
            RecommendationResult rec = recommendations.get(i);
            double score = rec.getScore() * weight;
            combinedScores.merge(rec.getProductId(), score, Double::sum);
        }
    }

    public static double calculateJaccardSimilarity(
            UserBehaviorRepository userBehaviorRepository,
            Long productId1, Long productId2) {
        List<Long> users1 = userBehaviorRepository.findUserIdsByProductId(productId1);
        List<Long> users2 = userBehaviorRepository.findUserIdsByProductId(productId2);
        Set<Long> set1 = new HashSet<>(users1);
        Set<Long> set2 = new HashSet<>(users2);
        Set<Long> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        Set<Long> union = new HashSet<>(set1);
        union.addAll(set2);
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }

    public static Map<Long, Double> createBehaviorVector(
            List<UserBehavior> behaviors,
            RecommendationAlgorithmProperties.Behavior behavior) {
        Map<Long, Double> vector = new HashMap<>();
        for (UserBehavior b : behaviors) {
            double weight = getBehaviorWeight(behavior, b.getBehaviorType());
            vector.merge(b.getProductId(), weight, Double::sum);
        }
        return vector;
    }

    public static double calculateCosineSimilarity(
            Map<Long, Double> vector1, Map<Long, Double> vector2) {
        Set<Long> commonProducts = new HashSet<>(vector1.keySet());
        commonProducts.retainAll(vector2.keySet());
        if (commonProducts.isEmpty()) return 0.0;
        double dotProduct = commonProducts.stream()
            .mapToDouble(productId -> vector1.get(productId) * vector2.get(productId))
            .sum();
        double mag1 = Math.sqrt(vector1.values().stream().mapToDouble(v -> v * v).sum());
        double mag2 = Math.sqrt(vector2.values().stream().mapToDouble(v -> v * v).sum());
        if (mag1 == 0.0 || mag2 == 0.0) return 0.0;
        return dotProduct / (mag1 * mag2);
    }
}
