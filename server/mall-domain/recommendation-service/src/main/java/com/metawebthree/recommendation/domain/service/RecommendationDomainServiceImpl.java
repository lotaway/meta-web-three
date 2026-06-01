package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.Recommendation.RecommendedItem;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

@Service
public class RecommendationDomainServiceImpl implements RecommendationDomainService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final Random random = new Random();

    public RecommendationDomainServiceImpl(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
    }

    @Override
    public Recommendation generateRecommendation(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        
        Recommendation recommendation = new Recommendation();
        recommendation.generate(userId, scene, algorithm);
        
        // Generate mock recommendations based on algorithm
        List<RecommendedItem> items = generateItems(userId, scene, algorithm, maxItems);
        recommendation.complete(items);
        
        // Calculate overall score
        BigDecimal avgScore = items.stream()
            .map(RecommendedItem::getScore)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(BigDecimal.valueOf(items.size()), 2, BigDecimal.ROUND_HALF_UP);
        recommendation.setScore(avgScore);
        
        return recommendationRepository.save(recommendation);
    }

    private List<RecommendedItem> generateItems(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        List<RecommendedItem> items = new ArrayList<>();
        
        for (int i = 0; i < maxItems; i++) {
            RecommendedItem item = new RecommendedItem();
            item.setSkuCode("SKU-" + (1000 + random.nextInt(9000)));
            item.setSkuName("Product " + (i + 1));
            item.setScore(BigDecimal.valueOf(100 - i * 5 - random.nextInt(10)));
            item.setRank(i + 1);
            item.setReason(getReason(algorithm, i));
            items.add(item);
        }
        
        return items;
    }

    private String getReason(Recommendation.RecommendationAlgorithm algorithm, int rank) {
        return switch (algorithm) {
            case COLLABORATIVE_FILTERING -> "Similar users also bought";
            case CONTENT_BASED -> "Based on your browsing history";
            case HYBRID -> "Recommended for you";
            case POPULARITY -> "Popular in your area";
            case AI_MODEL -> "AI personalized recommendation";
        };
    }

    @Override
    public void recordUserBehavior(Long userId, String skuCode, String behaviorType) {
        // Record user behavior for model training
        // In production, this would be stored in a behavior database
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
        
        // Apply filtering and boosting rules
        // In production, this would modify the recommendation items
    }

    @Override
    public Double calculateCTR(Long recommendationId) {
        Recommendation recommendation = recommendationRepository.findById(recommendationId)
            .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        
        if (recommendation.getStatus() == Recommendation.RecommendationStatus.CLICKED) {
            return 100.0; // Mock CTR calculation
        }
        return 0.0;
    }

    @Override
    public Double calculateConversionRate(Long recommendationId) {
        Recommendation recommendation = recommendationRepository.findById(recommendationId)
            .orElseThrow(() -> new IllegalArgumentException("Recommendation not found"));
        
        if (recommendation.getStatus() == Recommendation.RecommendationStatus.CONVERTED) {
            return 100.0; // Mock conversion rate calculation
        }
        return 0.0;
    }
}