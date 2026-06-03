package com.metawebthree.recommendation.application.query;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.domain.service.RecommendationDomainService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class RecommendationQueryService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final UserBehaviorRepository userBehaviorRepository;
    private final RecommendationResultRepository recommendationResultRepository;
    private final RecommendationDomainService recommendationDomainService;

    public RecommendationQueryService(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository,
            UserBehaviorRepository userBehaviorRepository,
            RecommendationResultRepository recommendationResultRepository,
            RecommendationDomainService recommendationDomainService) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
        this.userBehaviorRepository = userBehaviorRepository;
        this.recommendationResultRepository = recommendationResultRepository;
        this.recommendationDomainService = recommendationDomainService;
    }

    public Optional<Recommendation> getRecommendationById(Long id) {
        return recommendationRepository.findById(id);
    }

    public List<Recommendation> getUserRecommendations(Long userId) {
        return recommendationRepository.findByUserId(userId);
    }

    public List<Recommendation> getUserRecommendationsByScene(Long userId, String scene) {
        return recommendationRepository.findByUserIdAndScene(userId, scene);
    }

    public List<RecommendationRule> getRulesByScene(String scene) {
        return ruleRepository.findByScene(scene);
    }

    public List<RecommendationRule> getActiveRules(String scene) {
        return ruleRepository.findBySceneAndStatus(scene, RecommendationRule.RuleStatus.ACTIVE);
    }

    public Optional<RecommendationRule> getRuleById(Long id) {
        return ruleRepository.findById(id);
    }

    public List<RecommendationResult> getRecommendations(Long userId, int limit) {
        List<RecommendationResult> cached = recommendationResultRepository
            .findByUserIdOrderByScoreDesc(userId, LocalDateTime.now());
        if (!cached.isEmpty()) {
            return cached.stream().limit(limit).collect(Collectors.toList());
        }
        return recommendationDomainService.hybridRecommendation(userId, limit);
    }

    public List<RecommendationResult> getRecommendationsByAlgorithm(
            Long userId,
            RecommendationResult.RecommendationAlgorithm algorithm,
            int limit) {
        switch (algorithm) {
            case USER_BASED_CF:
                return recommendationDomainService.userBasedCollaborativeFiltering(userId, limit);
            case ITEM_BASED_CF:
                return recommendationDomainService.itemBasedCollaborativeFiltering(userId, limit);
            case CONTENT_BASED:
                return recommendationDomainService.contentBasedFiltering(userId, limit);
            case POPULARITY:
                return recommendationDomainService.popularityBasedRecommendation(userId, limit);
            default:
                return recommendationDomainService.hybridRecommendation(userId, limit);
        }
    }

    public List<UserBehavior> getUserBehaviorHistory(Long userId, int limit) {
        List<UserBehavior> behaviors = userBehaviorRepository
            .findByUserIdOrderByTimestampDesc(userId);
        if (limit > 0 && behaviors.size() > limit) {
            return behaviors.subList(0, limit);
        }
        return behaviors;
    }

    public List<ProductSimilarity> getProductSimilarity(Long productId, int limit) {
        return recommendationDomainService.itemBasedCollaborativeFiltering(productId, limit)
            .stream()
            .map(rec -> {
                ProductSimilarity ps = new ProductSimilarity();
                ps.setProductId1(productId);
                ps.setProductId2(rec.getProductId());
                ps.setSimilarityScore(rec.getScore());
                return ps;
            })
            .collect(Collectors.toList());
    }

    public RecommendationMetrics getRecommendationMetrics(Long userId) {
        List<RecommendationResult> recommendations = recommendationResultRepository
            .findByUserIdOrderByScoreDesc(userId, LocalDateTime.now());
        long totalRecommendations = recommendations.size();
        long clickedCount = recommendations.stream()
            .filter(RecommendationResult::getIsClicked)
            .count();
        long purchasedCount = recommendations.stream()
            .filter(RecommendationResult::getIsPurchased)
            .count();
        double clickThroughRate = totalRecommendations > 0 ? (double) clickedCount / totalRecommendations : 0.0;
        double conversionRate = totalRecommendations > 0 ? (double) purchasedCount / totalRecommendations : 0.0;
        return new RecommendationMetrics(totalRecommendations, clickedCount, purchasedCount, clickThroughRate, conversionRate);
    }

    public Page<RecommendationResult> getRecommendationsPaginated(Long userId, Pageable pageable) {
        List<RecommendationResult> allRecommendations = recommendationResultRepository
            .findByUserIdOrderByScoreDesc(userId, LocalDateTime.now());
        int start = (int) pageable.getOffset();
        int end = Math.min(start + pageable.getPageSize(), allRecommendations.size());
        List<RecommendationResult> pageContent = allRecommendations.subList(start, end);
        return new PageImpl<>(pageContent, pageable, allRecommendations.size());
    }

    public static class RecommendationMetrics {
        private final long totalRecommendations;
        private final long clickedCount;
        private final long purchasedCount;
        private final double clickThroughRate;
        private final double conversionRate;

        public RecommendationMetrics(long totalRecommendations, long clickedCount,
                                    long purchasedCount, double clickThroughRate,
                                    double conversionRate) {
            this.totalRecommendations = totalRecommendations;
            this.clickedCount = clickedCount;
            this.purchasedCount = purchasedCount;
            this.clickThroughRate = clickThroughRate;
            this.conversionRate = conversionRate;
        }

        public long getTotalRecommendations() { return totalRecommendations; }
        public long getClickedCount() { return clickedCount; }
        public long getPurchasedCount() { return purchasedCount; }
        public double getClickThroughRate() { return clickThroughRate; }
        public double getConversionRate() { return conversionRate; }
    }
}