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

@Service
public class RecommendationDomainServiceImpl implements RecommendationDomainService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CLICKS = new ConcurrentHashMap<>();
    private static final Map<Long, Map<String, Integer>> RECOMMENDATION_CONVERSIONS = new ConcurrentHashMap<>();

    public RecommendationDomainServiceImpl(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
        this.algorithmProperties = algorithmProperties;
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
            RecommendedItem item = new RecommendedItem();
            item.setSkuCode(generateSkuCode(userId, scene, i));
            item.setSkuName(generateSkuName(scene, i));
            
            double positionDecay = scoring.getScoreDecay() * i;
            double scoreVariation = calculateScoreVariation(userId, i);
            double algorithmScore = baseWeight * (100 - positionDecay + scoreVariation);
            
            item.setScore(BigDecimal.valueOf(algorithmScore)
                .setScale(2, RoundingMode.HALF_UP));
            item.setRank(i + 1);
            item.setReason(generateReason(algorithm, i, scene));
            items.add(item);
        }
        
        return items;
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
        
        if (clickCount == null || clickCount == 0) {
            return 0.0;
        }
        
        double ctr = (double) clickCount / impressionCount * 100;
        
        double highThreshold = algorithmProperties.getCtr().getHighThreshold();
        double lowThreshold = algorithmProperties.getCtr().getLowThreshold();
        
        if (ctr > highThreshold) {
            return Math.min(ctr, 100.0);
        } else if (ctr < lowThreshold) {
            return Math.max(ctr, 0.0);
        }
        
        return Math.round(ctr * 100.0) / 100.0;
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
        
        if (conversionCount == null || conversionCount == 0) {
            return 0.0;
        }
        
        double conversionRate = (double) conversionCount / clickCount * 100;
        
        double highThreshold = algorithmProperties.getConversion().getHighThreshold();
        double lowThreshold = algorithmProperties.getConversion().getLowThreshold();
        
        if (conversionRate > highThreshold) {
            return Math.min(conversionRate, 100.0);
        } else if (conversionRate < lowThreshold) {
            return Math.max(conversionRate, 0.0);
        }
        
        return Math.round(conversionRate * 100.0) / 100.0;
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
}