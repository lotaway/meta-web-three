package com.metawebthree.product_recommendation.domain.service;

import com.metawebthree.product_recommendation.domain.model.ProductProfile;
import com.metawebthree.product_recommendation.domain.model.Recommendation;
import com.metawebthree.product_recommendation.domain.model.RecommendationType;
import com.metawebthree.product_recommendation.domain.model.UserBehavior;
import com.metawebthree.product_recommendation.domain.repository.ProductProfileRepository;
import com.metawebthree.product_recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.product_recommendation.domain.repository.UserBehaviorRepository;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

public class RecommendationDomainService {

    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductProfileRepository productProfileRepository;
    private final RecommendationRepository recommendationRepository;

    public RecommendationDomainService(
            UserBehaviorRepository userBehaviorRepository,
            ProductProfileRepository productProfileRepository,
            RecommendationRepository recommendationRepository) {
        this.userBehaviorRepository = userBehaviorRepository;
        this.productProfileRepository = productProfileRepository;
        this.recommendationRepository = recommendationRepository;
    }

    public List<Recommendation> generateCollaborativeFilteringRecommendations(
            Long userId, int limit) {
        List<UserBehavior> userBehaviors = userBehaviorRepository.findByUserId(userId);
        if (userBehaviors.isEmpty()) {
            return Collections.emptyList();
        }

        Map<Long, Double> productScores = new HashMap<>();
        for (UserBehavior behavior : userBehaviors) {
            double weight = behavior.getWeight();
            double currentScore = productScores.getOrDefault(behavior.getProductId(), 0.0);
            productScores.put(behavior.getProductId(), currentScore + weight);
        }

        List<Map.Entry<Long, Double>> sortedProducts = productScores.entrySet().stream()
                .sorted(Map.Entry.<Long, Double>comparingByValue().reversed())
                .limit(limit * 3)
                .collect(Collectors.toList());

        Set<Long> excludedProducts = productScores.keySet();
        List<Recommendation> recommendations = new ArrayList<>();
        long id = System.currentTimeMillis();

        for (Map.Entry<Long, Double> entry : sortedProducts) {
            if (excludedProducts.contains(entry.getKey())) {
                continue;
            }
            if (recommendations.size() >= limit) {
                break;
            }

            Recommendation rec = new Recommendation(
                    id++,
                    userId,
                    entry.getKey(),
                    BigDecimal.valueOf(entry.getValue()).setScale(2, RoundingMode.HALF_UP),
                    RecommendationType.COLLABORATIVE_FILTERING,
                    "Users with similar behavior also liked"
            );
            recommendations.add(rec);
        }

        return recommendations;
    }

    public List<Recommendation> generateContentBasedRecommendations(
            Long userId, int limit) {
        List<UserBehavior> userBehaviors = userBehaviorRepository.findByUserId(userId);
        if (userBehaviors.isEmpty()) {
            return Collections.emptyList();
        }

        Map<Long, ProductProfile> viewedProfiles = new HashMap<>();
        for (UserBehavior behavior : userBehaviors) {
            ProductProfile profile = productProfileRepository.findByProductId(behavior.getProductId());
            if (profile != null) {
                viewedProfiles.put(behavior.getProductId(), profile);
            }
        }

        if (viewedProfiles.isEmpty()) {
            return Collections.emptyList();
        }

        Map<String, Double> userPreference = new HashMap<>();
        for (ProductProfile profile : viewedProfiles.values()) {
            if (profile.getTags() != null) {
                for (String tag : profile.getTags()) {
                    userPreference.merge(tag, 1.0, Double::sum);
                }
            }
            if (profile.getAttributes() != null) {
                for (String attr : profile.getAttributes()) {
                    userPreference.merge(attr, 0.5, Double::sum);
                }
            }
        }

        Map<Long, Double> productScores = new HashMap<>();
        List<ProductProfile> allProfiles = productProfileRepository.findByCategory(
                viewedProfiles.values().iterator().next().getCategory());

        for (ProductProfile profile : allProfiles) {
            if (viewedProfiles.containsKey(profile.getProductId())) {
                continue;
            }

            double score = 0.0;
            if (profile.getTags() != null) {
                for (String tag : profile.getTags()) {
                    score += userPreference.getOrDefault(tag, 0.0);
                }
            }
            if (profile.getAverageRating() != null) {
                score += profile.getAverageRating().doubleValue() * 2;
            }
            if (profile.getSalesCount() != null) {
                score += Math.log10(profile.getSalesCount() + 1);
            }

            if (score > 0) {
                productScores.put(profile.getProductId(), score);
            }
        }

        List<Map.Entry<Long, Double>> sortedProducts = productScores.entrySet().stream()
                .sorted(Map.Entry.<Long, Double>comparingByValue().reversed())
                .limit(limit)
                .collect(Collectors.toList());

        List<Recommendation> recommendations = new ArrayList<>();
        long id = System.currentTimeMillis();

        for (Map.Entry<Long, Double> entry : sortedProducts) {
            Recommendation rec = new Recommendation(
                    id++,
                    userId,
                    entry.getKey(),
                    BigDecimal.valueOf(entry.getValue()).setScale(2, RoundingMode.HALF_UP),
                    RecommendationType.CONTENT_BASED,
                    "Based on your interest in similar products"
            );
            recommendations.add(rec);
        }

        return recommendations;
    }

    public List<Recommendation> generateHybridRecommendations(
            Long userId, int limit) {
        List<Recommendation> collaborativeRecs = generateCollaborativeFilteringRecommendations(userId, limit);
        List<Recommendation> contentRecs = generateContentBasedRecommendations(userId, limit);

        Map<Long, Double> combinedScores = new HashMap<>();
        Map<Long, Recommendation> recMap = new HashMap<>();
        final long[] id = {System.currentTimeMillis()};

        double collabWeight = 0.6;
        double contentWeight = 0.4;

        for (Recommendation rec : collaborativeRecs) {
            double score = rec.getScore().doubleValue() * collabWeight;
            combinedScores.merge(rec.getProductId(), score, Double::sum);
            recMap.put(rec.getProductId(), rec);
        }

        for (Recommendation rec : contentRecs) {
            double score = rec.getScore().doubleValue() * contentWeight;
            combinedScores.merge(rec.getProductId(), score, Double::sum);
            recMap.put(rec.getProductId(), rec);
        }

        return combinedScores.entrySet().stream()
                .sorted(Map.Entry.<Long, Double>comparingByValue().reversed())
                .limit(limit)
                .map(entry -> {
                    Recommendation original = recMap.get(entry.getKey());
                    Recommendation hybrid = new Recommendation(
                            id[0]++,
                            userId,
                            entry.getKey(),
                            BigDecimal.valueOf(entry.getValue()).setScale(2, RoundingMode.HALF_UP),
                            RecommendationType.HYBRID,
                            "Combined collaborative and content-based recommendations"
                    );
                    hybrid.setCreatedAt(original.getCreatedAt());
                    return hybrid;
                })
                .collect(Collectors.toList());
    }

    public void saveRecommendations(List<Recommendation> recommendations) {
        if (recommendations.isEmpty()) {
            return;
        }
        recommendationRepository.deleteByUserId(recommendations.get(0).getUserId());
        recommendationRepository.batchSave(recommendations);
    }

    public void cleanupExpiredRecommendations() {
        recommendationRepository.deleteExpired();
    }
}