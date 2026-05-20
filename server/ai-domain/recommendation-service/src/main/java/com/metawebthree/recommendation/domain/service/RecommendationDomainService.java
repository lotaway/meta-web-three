package com.metawebthree.recommendation.domain.service;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import java.util.List;

public interface RecommendationDomainService {
    Recommendation generateRecommendation(Long userId, String scene, 
                                           Recommendation.RecommendationAlgorithm algorithm,
                                           int maxItems);
    
    void recordUserBehavior(Long userId, String skuCode, String behaviorType);
    
    List<Recommendation> getUserRecommendations(Long userId, String scene);
    
    RecommendationRule createRule(String ruleName, String scene, 
                                   RecommendationRule.RuleType type);
    
    void activateRule(Long ruleId);
    
    void applyRules(Recommendation recommendation);
    
    Double calculateCTR(Long recommendationId);
    
    Double calculateConversionRate(Long recommendationId);
}