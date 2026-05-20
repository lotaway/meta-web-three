package com.metawebthree.aiwarehouse.infrastructure.client;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import org.springframework.stereotype.Component;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class AIClientFactory {

    private final Map<String, AIClient> clients = new ConcurrentHashMap<>();

    public AIClient getClient(AICapability capability) {
        String key = capability.getCapabilityId();
        
        return clients.computeIfAbsent(key, k -> {
            return createClient(capability);
        });
    }

    private AIClient createClient(AICapability capability) {
        AICapability.AICapabilityType type = capability.getType();
        
        switch (type) {
            case FORECASTING:
                return new ForecastingClient(capability.getEndpoint(),
                    capability.getTimeoutMs(), capability.getRetryCount());
            case RECOMMENDATION:
                return new RecommendationClient(capability.getEndpoint(),
                    capability.getTimeoutMs(), capability.getRetryCount());
            case RISK_SCORING:
                return new RiskScoringClient(capability.getEndpoint(),
                    capability.getTimeoutMs(), capability.getRetryCount());
            default:
                return new GenericAIClient(capability.getEndpoint(),
                    capability.getTimeoutMs(), capability.getRetryCount());
        }
    }

    public void removeClient(String capabilityId) {
        clients.remove(capabilityId);
    }

    public void clear() {
        clients.clear();
    }
}