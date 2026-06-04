package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.Recommendation.RecommendedItem;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
@Service
public class RecommendationDomainServiceImpl implements RecommendationDomainService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;
    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductSimilarityRepository productSimilarityRepository;
    private final RecommendationResultRepository recommendationResultRepository;

    private final UserBasedCollaborativeFilteringService userBasedCFService;
    private final ItemBasedCollaborativeFilteringService itemBasedCFService;
    private final ContentBasedFilteringService contentBasedService;
    private final PopularityBasedRecommendationService popularityService;
    private final HybridRecommendationService hybridService;

    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CLICKS = new ConcurrentHashMap<>();
    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CONVERSIONS = new ConcurrentHashMap<>();

    public RecommendationDomainServiceImpl(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository,
            RecommendationAlgorithmProperties algorithmProperties,
            UserBehaviorRepository userBehaviorRepository,
            ProductSimilarityRepository productSimilarityRepository,
            RecommendationResultRepository recommendationResultRepository,
            UserBasedCollaborativeFilteringService userBasedCFService,
            ItemBasedCollaborativeFilteringService itemBasedCFService,
            ContentBasedFilteringService contentBasedService,
            PopularityBasedRecommendationService popularityService,
            HybridRecommendationService hybridService) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
        this.algorithmProperties = algorithmProperties;
        this.userBehaviorRepository = userBehaviorRepository;
        this.productSimilarityRepository = productSimilarityRepository;
        this.recommendationResultRepository = recommendationResultRepository;
        this.userBasedCFService = userBasedCFService;
        this.itemBasedCFService = itemBasedCFService;
        this.contentBasedService = contentBasedService;
        this.popularityService = popularityService;
        this.hybridService = hybridService;
    }

    @Override
    public Recommendation generateRecommendation(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        Recommendation recommendation = new Recommendation();
        recommendation.generate(userId, scene, algorithm);
        recommendation.setImpressionCount(0);
        recommendation.setClickCount(0);
        recommendation.setConversionCount(0);
        List<RecommendedItem> items = generateItemsByAlgorithm(userId, scene, algorithm, maxItems);
        recommendation.complete(items);
        BigDecimal avgScore = items.isEmpty() ? BigDecimal.ZERO : items.stream()
            .map(RecommendedItem::getScore)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(BigDecimal.valueOf(items.size()), 2, RoundingMode.HALF_UP);
        recommendation.setScore(avgScore);
        return recommendationRepository.save(recommendation);
    }

    private List<RecommendedItem> generateItemsByAlgorithm(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        RecommendationResult.RecommendationAlgorithm resultAlgo = toResultAlgorithm(algorithm);
        List<RecommendationResult> results = recommendationResultRepository
                .findByUserIdAndAlgorithm(userId, resultAlgo, LocalDateTime.now());
        return results.stream()
                .limit(maxItems)
                .map(this::toRecommendedItem)
                .collect(Collectors.toList());
    }

    private RecommendedItem toRecommendedItem(RecommendationResult result) {
        RecommendedItem item = new RecommendedItem();
        item.setSkuCode(String.valueOf(result.getProductId()));
        item.setScore(BigDecimal.valueOf(result.getScore()));
        item.setReason(result.getReason());
        item.setRank(0);
        return item;
    }

    private RecommendationResult.RecommendationAlgorithm toResultAlgorithm(
            Recommendation.RecommendationAlgorithm algorithm) {
        return switch (algorithm) {
            case COLLABORATIVE_FILTERING -> RecommendationResult.RecommendationAlgorithm.USER_BASED_CF;
            case CONTENT_BASED -> RecommendationResult.RecommendationAlgorithm.CONTENT_BASED;
            case HYBRID -> RecommendationResult.RecommendationAlgorithm.HYBRID;
            case POPULARITY -> RecommendationResult.RecommendationAlgorithm.POPULARITY;
            case AI_MODEL -> RecommendationResult.RecommendationAlgorithm.DEEP_LEARNING;
        };
    }

    @Override
    public void recordUserBehavior(Long userId, String skuCode, String behaviorType) {
        recommendationRepository.recordBehavior(userId, skuCode, behaviorType);
    }

    @Override
    public List<Recommendation> getUserRecommendations(Long userId, String scene) {
        return recommendationRepository.findByUserIdAndScene(userId, scene);
    }

    @Override
    public RecommendationRule createRule(String ruleName, String scene,
            RecommendationRule.RuleType type) {
        RecommendationRule rule = new RecommendationRule();
        rule.create(ruleName, scene, type);
        return ruleRepository.save(rule);
    }

    @Override
    public void activateRule(Long ruleId) {
        RecommendationRule rule = ruleRepository.findById(ruleId)
            .orElseThrow(() -> new IllegalArgumentException("Rule not found"));
        rule.activate();
        ruleRepository.update(rule);
    }

    @Override
    public void deleteRule(Long ruleId) {
        ruleRepository.deleteById(ruleId);
    }

    @Override
    public void applyRules(Recommendation recommendation) {
        List<RecommendationRule> activeRules = ruleRepository
            .findBySceneAndStatus(recommendation.getScene(), RecommendationRule.RuleStatus.ACTIVE);
        for (RecommendationRule rule : activeRules) {
            applyRuleToRecommendation(recommendation, rule);
        }
    }

    private void applyRuleToRecommendation(Recommendation recommendation, RecommendationRule rule) {
        if (recommendation.getItems() == null || recommendation.getItems().isEmpty()) {
            return;
        }
        List<RecommendedItem> boostedItems = recommendation.getItems().stream()
            .map(item -> {
                if (rule.getTargetSkus() != null && rule.getTargetSkus().contains(item.getSkuCode())) {
                    double boostFactor = rule.getBoostFactor() != null ? rule.getBoostFactor().doubleValue() : 1.0;
                    BigDecimal newScore = item.getScore().multiply(BigDecimal.valueOf(boostFactor));
                    item.setScore(newScore.setScale(2, RoundingMode.HALF_UP));
                }
                return item;
            })
            .collect(Collectors.toList());
        recommendation.setItems(boostedItems);
    }

    @Override
    public Double calculateCTR(Long recommendationId) {
        Recommendation recommendation = recommendationRepository.findById(recommendationId)
            .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        Integer impressionCount = recommendation.getImpressionCount();
        Integer clickCount = recommendation.getClickCount();
        if (impressionCount == null || impressionCount == 0) {
            return algorithmProperties.getCtr().getIndustryAverage();
        }
        if (clickCount == null || clickCount == 0) return 0.0;
        double ctr = (double) clickCount / impressionCount * 100;
        return clampThreshold(ctr, algorithmProperties.getCtr().getHighThreshold(), algorithmProperties.getCtr().getLowThreshold());
    }

    @Override
    public Double calculateConversionRate(Long recommendationId) {
        Recommendation recommendation = recommendationRepository.findById(recommendationId)
            .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        Integer clickCount = recommendation.getClickCount();
        Integer conversionCount = recommendation.getConversionCount();
        if (clickCount == null || clickCount == 0) {
            return algorithmProperties.getConversion().getIndustryAverage();
        }
        if (conversionCount == null || conversionCount == 0) return 0.0;
        double conversionRate = (double) conversionCount / clickCount * 100;
        return clampThreshold(conversionRate, algorithmProperties.getConversion().getHighThreshold(), algorithmProperties.getConversion().getLowThreshold());
    }

    private Double clampThreshold(double value, double highThreshold, double lowThreshold) {
        if (value > highThreshold) return Math.min(value, 100.0);
        if (value < lowThreshold) return Math.max(value, 0.0);
        return Math.round(value * 100.0) / 100.0;
    }

    public void recordClick(Long recommendationId, String skuCode) {
        RECOMMENDATION_CLICKS
            .computeIfAbsent(recommendationId, k -> new ConcurrentHashMap<>())
            .merge(skuCode, 1, Integer::sum);
    }

    public void recordConversion(Long recommendationId, String skuCode) {
        RECOMMENDATION_CONVERSIONS
            .computeIfAbsent(recommendationId, k -> new ConcurrentHashMap<>())
            .merge(skuCode, 1, Integer::sum);
    }

    @Override
    public List<RecommendationResult> userBasedCollaborativeFiltering(Long userId, int limit) {
        return userBasedCFService.userBasedCollaborativeFiltering(userId, limit);
    }

    @Override
    public List<RecommendationResult> itemBasedCollaborativeFiltering(Long userId, int limit) {
        return itemBasedCFService.itemBasedCollaborativeFiltering(userId, limit);
    }

    @Override
    public List<RecommendationResult> contentBasedFiltering(Long userId, int limit) {
        return contentBasedService.contentBasedFiltering(userId, limit);
    }

    @Override
    public List<RecommendationResult> popularityBasedRecommendation(Long userId, int limit) {
        return popularityService.popularityBasedRecommendation(userId, limit);
    }

    @Override
    public List<RecommendationResult> hybridRecommendation(Long userId, int limit) {
        return hybridService.hybridRecommendation(userId, limit);
    }

    @Override
    public double calculateProductSimilarity(Long productId1, Long productId2) {
        ProductSimilarity existing = productSimilarityRepository.findByProductIds(productId1, productId2);
        if (existing != null) {
            return existing.getSimilarityScore();
        }
        double similarity = RecommendationCalculationUtils.calculateJaccardSimilarity(
                userBehaviorRepository, productId1, productId2);
        ProductSimilarity productSimilarity = new ProductSimilarity();
        productSimilarity.setProductId1(productId1);
        productSimilarity.setProductId2(productId2);
        productSimilarity.setSimilarityScore(similarity);
        productSimilarity.setAlgorithm(ProductSimilarity.SimilarityAlgorithm.HYBRID);
        productSimilarity.setLastUpdated(LocalDateTime.now());
        productSimilarity.setUpdateCount(1);
        productSimilarityRepository.save(productSimilarity);
        return similarity;
    }

    @Override
    public double calculateUserSimilarity(Long userId1, Long userId2) {
        return userBasedCFService.calculateUserSimilarity(userId1, userId2);
    }

    @Override
    public void updateProductSimilarityMatrix() {
        log.warn("updateProductSimilarityMatrix not yet implemented");
    }

    @Override
    public void updateUserSimilarityMatrix() {
        log.warn("updateUserSimilarityMatrix not yet implemented");
    }
}
