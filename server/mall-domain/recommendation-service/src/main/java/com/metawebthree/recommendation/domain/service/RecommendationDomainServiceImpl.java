package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.Recommendation.RecommendedItem;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties.Scoring;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class RecommendationDomainServiceImpl implements RecommendationDomainService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CLICKS = new ConcurrentHashMap<>();
    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CONVERSIONS = new ConcurrentHashMap<>();

    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductSimilarityRepository productSimilarityRepository;
    private final RecommendationResultRepository recommendationResultRepository;

    private static final double USER_CF_WEIGHT = 0.3;
    private static final double ITEM_CF_WEIGHT = 0.3;
    private static final double CONTENT_WEIGHT = 0.2;
    private static final double POPULARITY_WEIGHT = 0.2;
    private static final double PURCHASE_BEHAVIOR_WEIGHT = 5.0;
    private static final double CART_BEHAVIOR_WEIGHT = 4.0;
    private static final double COLLECT_BEHAVIOR_WEIGHT = 3.0;
    private static final double CLICK_BEHAVIOR_WEIGHT = 2.0;
    private static final double VIEW_BEHAVIOR_WEIGHT = 1.0;
    private static final double DEFAULT_BEHAVIOR_WEIGHT = 1.0;
    private static final int SIMILAR_USER_MAX_COUNT = 20;
    private static final int RECOMMENDATION_EXPIRY_DAYS = 7;
    private static final String RECOMMENDED_BASED_ON = "Recommended based on ";
    private static final double POPULARITY_BASE_SCORE = 100.0;

    public RecommendationDomainServiceImpl(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository,
            RecommendationAlgorithmProperties algorithmProperties,
            UserBehaviorRepository userBehaviorRepository,
            ProductSimilarityRepository productSimilarityRepository,
            RecommendationResultRepository recommendationResultRepository) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
        this.algorithmProperties = algorithmProperties;
        this.userBehaviorRepository = userBehaviorRepository;
        this.productSimilarityRepository = productSimilarityRepository;
        this.recommendationResultRepository = recommendationResultRepository;
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
        
        BigDecimal avgScore = items.stream()
            .map(RecommendedItem::getScore)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(BigDecimal.valueOf(items.size()), 2, RoundingMode.HALF_UP);
        recommendation.setScore(avgScore);
        
        return recommendationRepository.save(recommendation);
    }

    private List<RecommendedItem> generateItemsByAlgorithm(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        List<RecommendedItem> items = new ArrayList<>();
        Scoring scoring = algorithmProperties.getScoring();
        double baseWeight = getAlgorithmWeight(algorithm, scoring);
        for (int i = 0; i < maxItems; i++) {
            items.add(createItem(userId, scene, algorithm, i, scoring, baseWeight));
        }
        return items;
    }

    private RecommendedItem createItem(Long userId, String scene, Recommendation.RecommendationAlgorithm algorithm,
                                        int i, Scoring scoring, double baseWeight) {
        RecommendedItem item = new RecommendedItem();
        item.setSkuCode(generateSkuCode(userId, scene, i));
        item.setSkuName(generateSkuName(scene, i));
        double positionDecay = scoring.getScoreDecay() * i;
        double scoreVariation = calculateScoreVariation(userId, i);
        double algorithmScore = baseWeight * (100 - positionDecay + scoreVariation);
        item.setScore(BigDecimal.valueOf(algorithmScore).setScale(2, RoundingMode.HALF_UP));
        item.setRank(i + 1);
        item.setReason(generateReason(algorithm, i, scene));
        return item;
    }

    private double getAlgorithmWeight(Recommendation.RecommendationAlgorithm algorithm, Scoring scoring) {
        return switch (algorithm) {
            case COLLABORATIVE_FILTERING -> scoring.getCollaborativeWeight();
            case CONTENT_BASED -> scoring.getContentWeight();
            case HYBRID -> scoring.getHybridWeight();
            case POPULARITY -> scoring.getPopularityWeight();
            case AI_MODEL -> scoring.getAiModelWeight();
        };
    }

    private String generateSkuCode(Long userId, String scene, int index) {
        long seed = userId + scene.hashCode() + index * 31;
        int suffix = Math.abs((int)(seed % 10000));
        return String.format("SKU-%04d-%s", suffix, scene.substring(0, Math.min(3, scene.length())).toUpperCase());
    }

    private String generateSkuName(String scene, int index) {
        String[] productTypes = {"Wireless Headphones", "Smart Watch", "Laptop Stand", 
            "USB-C Hub", "Mechanical Keyboard", "Gaming Mouse", "Monitor Light",
            "Webcam HD", "Portable SSD", "Desk Mat"};
        int typeIndex = Math.abs((scene.hashCode() + index) % productTypes.length);
        return productTypes[typeIndex] + " Pro " + (index + 1);
    }

    private double calculateScoreVariation(Long userId, int position) {
        long seed = userId * 31L + position * 17L;
        return (seed % 20) - 10;
    }

    private String generateReason(Recommendation.RecommendationAlgorithm algorithm, int rank, String scene) {
        return switch (algorithm) {
            case COLLABORATIVE_FILTERING -> "Users with similar preferences also chose this item";
            case CONTENT_BASED -> "Based on your interest in " + scene + " category";
            case HYBRID -> "Trending in " + scene + " - high compatibility with your profile";
            case POPULARITY -> "Top seller in " + scene + " - " + (rank + 1) + " units sold today";
            case AI_MODEL -> "AI analysis indicates high relevance to your browsing pattern";
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
        List<Long> similarUserIds = findSimilarUsers(userId, SIMILAR_USER_MAX_COUNT);
        Map<Long, Double> productScores = buildUserBasedProductScores(userId, similarUserIds);
        return productScores.entrySet().stream()
            .map(entry -> createRecommendationResult(userId, entry.getKey(),
                   entry.getValue(), RecommendationResult.RecommendationAlgorithm.USER_BASED_CF))
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

    @Override
    public List<RecommendationResult> itemBasedCollaborativeFiltering(Long userId, int limit) {
        Map<Long, Double> productScores = buildItemBasedProductScores(userId);
        return productScores.entrySet().stream()
            .map(entry -> createRecommendationResult(userId, entry.getKey(),
                   entry.getValue(), RecommendationResult.RecommendationAlgorithm.ITEM_BASED_CF))
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
        double behaviorWeight = getBehaviorWeight(behavior.getBehaviorType());
        List<ProductSimilarity> similarProducts = productSimilarityRepository.findSimilarProducts(productId);
        for (ProductSimilarity similarity : similarProducts) {
            Long similarProductId = similarity.getProductId1().equals(productId)
                ? similarity.getProductId2()
                : similarity.getProductId1();
            double score = similarity.getSimilarityScore() * behaviorWeight;
            productScores.merge(similarProductId, score, Double::sum);
        }
    }

    @Override
    public List<RecommendationResult> contentBasedFiltering(Long userId, int limit) {
        Map<Long, Double> productScores = buildContentBasedProductScores(userId);
        return productScores.entrySet().stream()
            .map(entry -> createRecommendationResult(userId, entry.getKey(),
                   entry.getValue(), RecommendationResult.RecommendationAlgorithm.CONTENT_BASED))
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

    @Override
    public List<RecommendationResult> popularityBasedRecommendation(Long userId, int limit) {
        List<RecommendationResult> recommendations = new ArrayList<>();
        for (long i = 1; i <= limit; i++) {
            RecommendationResult result = createRecommendationResult(
                userId, i,
                POPULARITY_BASE_SCORE - i,
                RecommendationResult.RecommendationAlgorithm.POPULARITY);
            recommendations.add(result);
        }
        return recommendations;
    }

    @Override
    public List<RecommendationResult> hybridRecommendation(Long userId, int limit) {
        Map<Long, Double> combinedScores = buildHybridCombinedScores(userId, limit);
        return combinedScores.entrySet().stream()
            .map(entry -> createRecommendationResult(userId, entry.getKey(),
                   entry.getValue(), RecommendationResult.RecommendationAlgorithm.HYBRID))
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private Map<Long, Double> buildHybridCombinedScores(Long userId, int limit) {
        List<RecommendationResult> userCfRecs = userBasedCollaborativeFiltering(userId, limit * 2);
        List<RecommendationResult> itemCfRecs = itemBasedCollaborativeFiltering(userId, limit * 2);
        List<RecommendationResult> contentRecs = contentBasedFiltering(userId, limit * 2);
        List<RecommendationResult> popularityRecs = popularityBasedRecommendation(userId, limit * 2);
        Map<Long, Double> combinedScores = new HashMap<>();
        combineRecommendations(combinedScores, userCfRecs, USER_CF_WEIGHT);
        combineRecommendations(combinedScores, itemCfRecs, ITEM_CF_WEIGHT);
        combineRecommendations(combinedScores, contentRecs, CONTENT_WEIGHT);
        combineRecommendations(combinedScores, popularityRecs, POPULARITY_WEIGHT);
        return combinedScores;
    }

    @Override
    public double calculateProductSimilarity(Long productId1, Long productId2) {
        ProductSimilarity existing = productSimilarityRepository.findByProductIds(productId1, productId2);
        if (existing != null) {
            return existing.getSimilarityScore();
        }
        double similarity = calculateJaccardSimilarity(productId1, productId2);
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

    private double calculateJaccardSimilarity(Long productId1, Long productId2) {
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

    @Override
    public double calculateUserSimilarity(Long userId1, Long userId2) {
        List<UserBehavior> behaviors1 = userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId1);
        List<UserBehavior> behaviors2 = userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId2);
        Map<Long, Double> vector1 = createBehaviorVector(behaviors1);
        Map<Long, Double> vector2 = createBehaviorVector(behaviors2);
        return calculateCosineSimilarity(vector1, vector2);
    }

    @Override
    public void updateProductSimilarityMatrix() {
    }

    @Override
    public void updateUserSimilarityMatrix() {
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

    private double getBehaviorWeight(UserBehavior.BehaviorType behaviorType) {
        switch (behaviorType) {
            case PURCHASE: return PURCHASE_BEHAVIOR_WEIGHT;
            case CART: return CART_BEHAVIOR_WEIGHT;
            case COLLECT: return COLLECT_BEHAVIOR_WEIGHT;
            case CLICK: return CLICK_BEHAVIOR_WEIGHT;
            case VIEW: return VIEW_BEHAVIOR_WEIGHT;
            default: return DEFAULT_BEHAVIOR_WEIGHT;
        }
    }

    private Map<Long, Double> createBehaviorVector(List<UserBehavior> behaviors) {
        Map<Long, Double> vector = new HashMap<>();
        for (UserBehavior behavior : behaviors) {
            double weight = getBehaviorWeight(behavior.getBehaviorType());
            vector.merge(behavior.getProductId(), weight, Double::sum);
        }
        return vector;
    }

    private double calculateCosineSimilarity(Map<Long, Double> vector1, Map<Long, Double> vector2) {
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

    private void combineRecommendations(Map<Long, Double> combinedScores,
                                        List<RecommendationResult> recommendations,
                                        double weight) {
        for (int i = 0; i < recommendations.size(); i++) {
            RecommendationResult rec = recommendations.get(i);
            double score = rec.getScore() * weight;
            combinedScores.merge(rec.getProductId(), score, Double::sum);
        }
    }

    private RecommendationResult createRecommendationResult(Long userId, Long productId,
                                                          Double score,
                                                          RecommendationResult.RecommendationAlgorithm algorithm) {
        RecommendationResult result = new RecommendationResult();
        result.setUserId(userId);
        result.setProductId(productId);
        result.setScore(score);
        result.setAlgorithm(algorithm);
        result.setReason(RECOMMENDED_BASED_ON + algorithm.name());
        result.setCreatedAt(LocalDateTime.now());
        result.setExpiresAt(LocalDateTime.now().plusDays(RECOMMENDATION_EXPIRY_DAYS));
        result.setIsClicked(false);
        result.setIsPurchased(false);
        return result;
    }
}