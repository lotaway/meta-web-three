package com.metawebthree.recommendation.infrastructure.event;

import com.metawebthree.common.event.DomainEventPublisher;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class RecommendationEventPublisher {

    private static final String EVENT_RECOMMENDATION_GENERATED = "recommendation.generated";
    private static final String EVENT_BEHAVIOR_RECORDED = "recommendation.behavior.recorded";
    private static final String EVENT_RULE_CREATED = "recommendation.rule.created";
    private static final String EVENT_RULE_ACTIVATED = "recommendation.rule.activated";

    private final DomainEventPublisher domainEventPublisher;

    public RecommendationEventPublisher(DomainEventPublisher domainEventPublisher) {
        this.domainEventPublisher = domainEventPublisher;
    }

    public void publishRecommendationGenerated(Long recommendationId, Long userId, String scene) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", recommendationId);
        data.put("userId", userId);
        data.put("scene", scene);
        data.put("eventType", EVENT_RECOMMENDATION_GENERATED);
        domainEventPublisher.publish(EVENT_RECOMMENDATION_GENERATED, data);
    }

    public void publishUserBehaviorRecorded(Long userId, String skuCode, String behaviorType) {
        Map<String, Object> data = new HashMap<>();
        data.put("userId", userId);
        data.put("skuCode", skuCode);
        data.put("behaviorType", behaviorType);
        data.put("eventType", EVENT_BEHAVIOR_RECORDED);
        domainEventPublisher.publish(EVENT_BEHAVIOR_RECORDED, data);
    }

    public void publishRuleCreated(Long ruleId, String ruleName, String scene) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", ruleId);
        data.put("ruleName", ruleName);
        data.put("scene", scene);
        data.put("eventType", EVENT_RULE_CREATED);
        domainEventPublisher.publish(EVENT_RULE_CREATED, data);
    }

    public void publishRuleActivated(Long ruleId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", ruleId);
        data.put("eventType", EVENT_RULE_ACTIVATED);
        domainEventPublisher.publish(EVENT_RULE_ACTIVATED, data);
    }
}
