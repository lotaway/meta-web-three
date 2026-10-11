package com.metawebthree.recommendation.application.command;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.domain.service.RecommendationDomainService;
import com.metawebthree.recommendation.infrastructure.event.RecommendationEventPublisher;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class RecommendationCommandService {

    private final RecommendationDomainService domainService;
    private final RecommendationEventPublisher eventPublisher;
    private final UserBehaviorRepository userBehaviorRepository;
    private final RecommendationResultRepository recommendationResultRepository;
    private final RecommendationRuleRepository ruleRepository;

    public RecommendationCommandService(
            RecommendationDomainService domainService,
            RecommendationEventPublisher eventPublisher,
            UserBehaviorRepository userBehaviorRepository,
            RecommendationResultRepository recommendationResultRepository,
            RecommendationRuleRepository ruleRepository) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
        this.userBehaviorRepository = userBehaviorRepository;
        this.recommendationResultRepository = recommendationResultRepository;
        this.ruleRepository = ruleRepository;
    }

    public Recommendation generateRecommendation(Long userId, String scene,
            Recommendation.RecommendationAlgorithm algorithm, int maxItems) {

        Recommendation recommendation = domainService.generateRecommendation(
            userId, scene, algorithm, maxItems);

        eventPublisher.publishRecommendationGenerated(
            recommendation.getId(), userId, scene);

        return recommendation;
    }

    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        domainService.recordUserBehavior(userId, skuCode, behaviorType);
        eventPublisher.publishUserBehaviorRecorded(userId, skuCode, behaviorType);
    }

    public RecommendationRule createRule(String ruleName, String scene, RecommendationRule.RuleType type) {
        RecommendationRule rule = domainService.createRule(ruleName, scene, type);
        eventPublisher.publishRuleCreated(rule.getId(), ruleName, scene);
        return rule;
    }

    public void activateRule(Long ruleId) {
        domainService.activateRule(ruleId);
        eventPublisher.publishRuleActivated(ruleId);
    }

    public void deleteRule(Long ruleId) {
        domainService.deleteRule(ruleId);
    }

    public void recordClick(Long recommendationId, String skuCode) {
        recommendationResultRepository.markAsClicked(recommendationId);
    }

    public void recordConversion(Long recommendationId, String skuCode) {
        recommendationResultRepository.markAsPurchased(recommendationId);
    }

    public UserBehavior recordUserBehavior(Long userId, Long productId,
                                            UserBehavior.BehaviorType behaviorType,
                                            Double behaviorValue, String sessionId, String source) {
        UserBehavior behavior = new UserBehavior();
        behavior.setUserId(userId);
        behavior.setProductId(productId);
        behavior.setBehaviorType(behaviorType);
        behavior.setBehaviorValue(behaviorValue != null ? behaviorValue : getDefaultBehaviorValue(behaviorType));
        behavior.setTimestamp(LocalDateTime.now());
        behavior.setSessionId(sessionId);
        behavior.setSource(source);
        return userBehaviorRepository.save(behavior);
    }

    public List<RecommendationResult> generateRecommendationsByAlgorithm(
            Long userId,
            RecommendationResult.RecommendationAlgorithm algorithm,
            int limit) {
        List<RecommendationResult> recommendations = getRecommendationsByAlgorithm(userId, algorithm, limit);
        for (int i = 0; i < recommendations.size(); i++) {
            RecommendationResult rec = recommendations.get(i);
            rec.setPosition(i + 1);
            recommendationResultRepository.save(rec);
        }
        return recommendations;
    }

    private List<RecommendationResult> getRecommendationsByAlgorithm(
            Long userId,
            RecommendationResult.RecommendationAlgorithm algorithm,
            int limit) {
        switch (algorithm) {
            case USER_BASED_CF:
                return domainService.userBasedCollaborativeFiltering(userId, limit);
            case ITEM_BASED_CF:
                return domainService.itemBasedCollaborativeFiltering(userId, limit);
            case CONTENT_BASED:
                return domainService.contentBasedFiltering(userId, limit);
            case POPULARITY:
                return domainService.popularityBasedRecommendation(userId, limit);
            default:
                return domainService.hybridRecommendation(userId, limit);
        }
    }

    public void markRecommendationClicked(Long recommendationId) {
        recommendationResultRepository.markAsClicked(recommendationId);
    }

    public void markRecommendationPurchased(Long recommendationId) {
        recommendationResultRepository.markAsPurchased(recommendationId);
    }

    public void markPurchasedByProduct(Long userId, Long productId) {
        List<RecommendationResult> results = recommendationResultRepository.findByUserIdAndProductId(userId, productId);
        for (RecommendationResult result : results) {
            result.setIsPurchased(true);
            recommendationResultRepository.save(result);
        }
    }

    public void markPurchasedByProducts(Long userId, List<Long> productIds) {
        recommendationResultRepository.markPurchasedByUserIdAndProductIds(userId, productIds);
    }

    public void updateProductSimilarities() {
        domainService.updateProductSimilarityMatrix();
    }

    public void cleanupOldData(int daysToKeep) {
        LocalDateTime cutoff = LocalDateTime.now().minusDays(daysToKeep);
        userBehaviorRepository.deleteByTimestampBefore(cutoff);
        recommendationResultRepository.deleteByExpiresAtBefore(LocalDateTime.now());
    }

    private Double getDefaultBehaviorValue(UserBehavior.BehaviorType behaviorType) {
        switch (behaviorType) {
            case VIEW: return 1.0;
            case CLICK: return 2.0;
            case CART: return 4.0;
            case PURCHASE: return 5.0;
            case COLLECT: return 3.0;
            case SHARE: return 3.0;
            case REVIEW: return 3.0;
            case SEARCH: return 1.0;
            default: return 1.0;
        }
    }
}
