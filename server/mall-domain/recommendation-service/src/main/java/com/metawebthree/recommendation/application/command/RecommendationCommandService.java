package com.metawebthree.recommendation.application.command;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.service.RecommendationDomainService;
import com.metawebthree.recommendation.infrastructure.event.RecommendationEventPublisher;
import org.springframework.stereotype.Service;

@Service
public class RecommendationCommandService {

    private final RecommendationDomainService domainService;
    private final RecommendationEventPublisher eventPublisher;

    public RecommendationCommandService(
            RecommendationDomainService domainService,
            RecommendationEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
    }

    public Long generateRecommendation(Long userId, String scene, 
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {
        
        Recommendation recommendation = domainService.generateRecommendation(
            userId, scene, algorithm, maxItems);
        
        eventPublisher.publishRecommendationGenerated(
            recommendation.getId(), userId, scene);
        
        return recommendation.getId();
    }

    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        domainService.recordUserBehavior(userId, skuCode, behaviorType);
        eventPublisher.publishUserBehaviorRecorded(userId, skuCode, behaviorType);
    }

    public Long createRule(String ruleName, String scene, RecommendationRule.RuleType type) {
        RecommendationRule rule = domainService.createRule(ruleName, scene, type);
        eventPublisher.publishRuleCreated(rule.getId(), ruleName, scene);
        return rule.getId();
    }

    public void activateRule(Long ruleId) {
        domainService.activateRule(ruleId);
        eventPublisher.publishRuleActivated(ruleId);
    }

    public void recordClick(Long recommendationId, String skuCode) {
        // Implementation
    }

    public void recordConversion(Long recommendationId, String skuCode) {
        // Implementation
    }
}