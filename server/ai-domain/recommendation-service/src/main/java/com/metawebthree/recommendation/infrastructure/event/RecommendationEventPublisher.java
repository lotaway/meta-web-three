package com.metawebthree.recommendation.infrastructure.event;

import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class RecommendationEventPublisher {

    public void publishRecommendationGenerated(Long recommendationId, Long userId, String scene) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RECOMMENDATION_GENERATED");
        event.put("recommendationId", recommendationId);
        event.put("userId", userId);
        event.put("scene", scene);
    }

    public void publishUserBehaviorRecorded(Long userId, String skuCode, String behaviorType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "USER_BEHAVIOR_RECORDED");
        event.put("userId", userId);
        event.put("skuCode", skuCode);
        event.put("behaviorType", behaviorType);
    }

    public void publishRuleCreated(Long ruleId, String ruleName, String scene) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RULE_CREATED");
        event.put("ruleId", ruleId);
        event.put("ruleName", ruleName);
        event.put("scene", scene);
    }

    public void publishRuleActivated(Long ruleId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RULE_ACTIVATED");
        event.put("ruleId", ruleId);
    }

    public void publishRecommendationClicked(Long recommendationId, String skuCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RECOMMENDATION_CLICKED");
        event.put("recommendationId", recommendationId);
        event.put("skuCode", skuCode);
    }

    public void publishRecommendationConverted(Long recommendationId, String skuCode) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "RECOMMENDATION_CONVERTED");
        event.put("recommendationId", recommendationId);
        event.put("skuCode", skuCode);
    }
}