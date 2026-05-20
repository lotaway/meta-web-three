package com.metawebthree.aiwarehouse.infrastructure.client;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class FallbackHandler {

    private final Map<String, FallbackStrategy> strategies = new HashMap<>();

    public FallbackHandler() {
        strategies.put("ALGORITHM", new AlgorithmFallbackStrategy());
        strategies.put("HUMAN", new HumanFallbackStrategy());
        strategies.put("DEFAULT_VALUE", new DefaultValueFallbackStrategy());
        strategies.put("CACHE", new CacheFallbackStrategy());
        strategies.put("NONE", new NoFallbackStrategy());
    }

    public FallbackResponse handleFallback(AICapability capability, AIRequest originalRequest) {
        String fallbackType = capability.getFallbackType();
        FallbackStrategy strategy = strategies.getOrDefault(fallbackType, 
            new NoFallbackStrategy());
        
        return strategy.execute(capability, originalRequest);
    }

    public interface FallbackStrategy {
        FallbackResponse execute(AICapability capability, AIRequest request);
    }

    public static class FallbackResponse {
        private final boolean success;
        private final String data;
        private final String reason;
        private final String fallbackType;

        public FallbackResponse(boolean success, String data, String reason, 
                               String fallbackType) {
            this.success = success;
            this.data = data;
            this.reason = reason;
            this.fallbackType = fallbackType;
        }

        public static FallbackResponse success(String data, String reason, 
                                               String fallbackType) {
            return new FallbackResponse(true, data, reason, fallbackType);
        }

        public static FallbackResponse failure(String reason) {
            return new FallbackResponse(false, null, reason, "NONE");
        }

        public boolean isSuccess() { return success; }
        public String getData() { return data; }
        public String getReason() { return reason; }
        public String getFallbackType() { return fallbackType; }
    }

    private static class AlgorithmFallbackStrategy implements FallbackStrategy {
        @Override
        public FallbackResponse execute(AICapability capability, AIRequest request) {
            return FallbackResponse.success(
                "{\"algorithm\":\"fallback\",\"result\":\"using_rule_based_result\"}",
                "AI service unavailable, using algorithm fallback",
                "ALGORITHM"
            );
        }
    }

    private static class HumanFallbackStrategy implements FallbackStrategy {
        @Override
        public FallbackResponse execute(AICapability capability, AIRequest request) {
            return FallbackResponse.success(
                "{\"status\":\"pending_human_review\",\"ticketId\":\"\"}",
                "AI service unavailable, human review required",
                "HUMAN"
            );
        }
    }

    private static class DefaultValueFallbackStrategy implements FallbackStrategy {
        @Override
        public FallbackResponse execute(AICapability capability, AIRequest request) {
            return FallbackResponse.success(
                "{\"result\":\"default_value\",\"source\":\"fallback\"}",
                "AI service unavailable, using default value",
                "DEFAULT_VALUE"
            );
        }
    }

    private static class CacheFallbackStrategy implements FallbackStrategy {
        @Override
        public FallbackResponse execute(AICapability capability, AIRequest request) {
            return FallbackResponse.success(
                "{\"result\":\"cache_miss\",\"source\":\"fallback\"}",
                "AI service unavailable, cache miss",
                "CACHE"
            );
        }
    }

    private static class NoFallbackStrategy implements FallbackStrategy {
        @Override
        public FallbackResponse execute(AICapability capability, AIRequest request) {
            return FallbackResponse.failure("No fallback available");
        }
    }
}