package com.metawebthree.aiwarehouse.infrastructure.router;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.client.AIClient;
import com.metawebthree.aiwarehouse.infrastructure.client.AIClientFactory;
import com.metawebthree.aiwarehouse.infrastructure.client.AIRequest;
import com.metawebthree.aiwarehouse.infrastructure.client.AIResponse;
import com.metawebthree.aiwarehouse.infrastructure.client.FallbackHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
public class FallbackRouter {

    private static final Logger log = LoggerFactory.getLogger(FallbackRouter.class);

    private final AIClientFactory clientFactory;
    private final FallbackHandler fallbackHandler;
    private final Map<WarehouseCapability, AlgorithmFallback> algorithmFallbacks;

    public FallbackRouter(AIClientFactory clientFactory,
            FallbackHandler fallbackHandler,
            List<AlgorithmFallback> fallbackImpls) {
        this.clientFactory = clientFactory;
        this.fallbackHandler = fallbackHandler;
        this.algorithmFallbacks = fallbackImpls.stream()
            .collect(Collectors.toMap(AlgorithmFallback::getCapability, Function.identity()));
    }

    public RouteResult route(WarehouseCapability capability, AIRequest request) {
        AICapability aiCapability = buildCapability(capability);
        
        return tryAI(aiCapability, request, capability);
    }

    private RouteResult tryAI(AICapability capability, AIRequest request,
            WarehouseCapability warehouseCap) {
        try {
            AIClient client = clientFactory.getClient(capability);
            if (client != null && client.isAvailable()) {
                AIResponse response = client.invoke(request);
                if (response != null && response.isSuccess()) {
                    log.info("AI route succeeded for capability: {}", capability.getCapabilityId());
                    return RouteResult.success(response.getData(), RouteType.AI);
                }
            }
        } catch (Exception e) {
            log.warn("AI invocation failed for {}, trying algorithm fallback: {}",
                capability.getCapabilityId(), e.getMessage());
        }
        
        return tryAlgorithm(capability, request, warehouseCap);
    }

    private RouteResult tryAlgorithm(AICapability capability, AIRequest request,
            WarehouseCapability warehouseCap) {
        AlgorithmFallback fallback = algorithmFallbacks.get(warehouseCap);
        if (fallback != null) {
            try {
                Object result = fallback.execute(request.getPayload());
                log.info("Algorithm fallback succeeded for capability: {}",
                    capability.getCapabilityId());
                return RouteResult.success(serializeResult(result), RouteType.ALGORITHM);
            } catch (Exception e) {
                log.warn("Algorithm fallback failed for {}, trying human fallback: {}",
                    capability.getCapabilityId(), e.getMessage());
            }
        }
        
        return tryHuman(capability, request);
    }

    private RouteResult tryHuman(AICapability capability, AIRequest request) {
        FallbackHandler.FallbackResponse response = fallbackHandler.handleFallback(
            capability, request);
        
        if (response.isSuccess()) {
            log.info("Human fallback succeeded for capability: {}",
                capability.getCapabilityId());
            return RouteResult.success(response.getData(), RouteType.HUMAN);
        }
        
        log.error("All fallback routes exhausted for capability: {}",
            capability.getCapabilityId());
        return RouteResult.failure("All routes failed: AI unavailable, algorithm failed, "
            + "human ticket creation failed");
    }

    private AICapability buildCapability(WarehouseCapability capability) {
        AICapability aiCap = new AICapability();
        aiCap.setCapabilityId(capability.getCapabilityId());
        aiCap.setCapabilityName(capability.getCapabilityName());
        aiCap.setFallbackType(capability.getDefaultFallbackType().name());
        return aiCap;
    }

    private String serializeResult(Object result) {
        if (result instanceof String) {
            return (String) result;
        }
        return result != null ? result.toString() : "{}";
    }

    public enum RouteType {
        AI, ALGORITHM, HUMAN
    }

    public static class RouteResult {
        private final boolean success;
        private final String data;
        private final RouteType routeType;
        private final String error;

        private RouteResult(boolean success, String data, RouteType routeType,
                String error) {
            this.success = success;
            this.data = data;
            this.routeType = routeType;
            this.error = error;
        }

        public static RouteResult success(String data, RouteType routeType) {
            return new RouteResult(true, data, routeType, null);
        }

        public static RouteResult failure(String error) {
            return new RouteResult(false, null, null, error);
        }

        public boolean isSuccess() { return success; }
        public String getData() { return data; }
        public RouteType getRouteType() { return routeType; }
        public String getError() { return error; }
    }

    public interface AlgorithmFallback {
        WarehouseCapability getCapability();
        Object execute(String payload);
    }
}