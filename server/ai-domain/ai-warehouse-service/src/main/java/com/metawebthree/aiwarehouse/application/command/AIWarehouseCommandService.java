package com.metawebthree.aiwarehouse.application.command;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import com.metawebthree.aiwarehouse.domain.service.AIWarehouseDomainService;
import com.metawebthree.aiwarehouse.infrastructure.client.AIClient;
import com.metawebthree.aiwarehouse.infrastructure.client.AIClientFactory;
import com.metawebthree.aiwarehouse.infrastructure.client.AIRequest;
import com.metawebthree.aiwarehouse.infrastructure.client.AIResponse;
import com.metawebthree.aiwarehouse.infrastructure.client.FallbackHandler;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class AIWarehouseCommandService {

    private final AIWarehouseDomainService domainService;
    private final AIClientFactory clientFactory;
    private final FallbackHandler fallbackHandler;

    public AIWarehouseCommandService(
            AIWarehouseDomainService domainService,
            AIClientFactory clientFactory,
            FallbackHandler fallbackHandler) {
        this.domainService = domainService;
        this.clientFactory = clientFactory;
        this.fallbackHandler = fallbackHandler;
    }

    public AICapability registerCapability(String capabilityId, String capabilityName,
            String type, String endpoint, String fallbackType, String fallbackConfig) {
        
        AICapability.AICapabilityType capabilityType = AICapability.AICapabilityType.valueOf(
            type.toUpperCase());
        AICapability.FallbackType fbType = fallbackType != null ? 
            AICapability.FallbackType.valueOf(fallbackType.toUpperCase()) : 
            AICapability.FallbackType.NONE;
        
        return domainService.registerCapability(
            capabilityId, capabilityName, capabilityType, endpoint, fbType, fallbackConfig);
    }

    public void updateCapability(String capabilityId, String endpoint,
                                 String fallbackType, String fallbackConfig) {
        AICapability.FallbackType fbType = fallbackType != null ?
            AICapability.FallbackType.valueOf(fallbackType.toUpperCase()) : null;
        domainService.updateCapability(capabilityId, endpoint, fbType, fallbackConfig);
    }

    public void enableCapability(String capabilityId) {
        domainService.enableCapability(capabilityId);
        clientFactory.removeClient(capabilityId);
    }

    public void removeCapability(String capabilityId) {
        domainService.removeCapability(capabilityId);
        clientFactory.removeClient(capabilityId);
    }

    public void disableCapability(String capabilityId) {
        domainService.disableCapability(capabilityId);
        clientFactory.removeClient(capabilityId);
    }

    public AIInvokeResult invokeAI(String capabilityId, String scene, 
                                   Long callerServiceId, String callerServiceName,
                                   String requestPayload) {
        
        AICapability capability = domainService.getCapability(capabilityId)
            .orElseThrow(() -> new IllegalArgumentException("Capability not found: " + capabilityId));
        
        if (!capability.isAvailable()) {
            return handleFallback(capability, scene, callerServiceId, 
                                 callerServiceName, requestPayload, "Capability disabled");
        }
        
        AIRequestRecord record = domainService.createRequestRecord(
            capabilityId, capability.getCapabilityName(), scene,
            callerServiceId, callerServiceName, requestPayload);
        
        try {
            AIClient client = clientFactory.getClient(capability);
            AIRequest request = new AIRequest(capabilityId, requestPayload);
            
            record.startProcessing();
            AIResponse response = client.isAvailable() ?
                client.invoke(request) : null;
            
            if (response != null && response.isSuccess()) {
                domainService.markRequestSuccess(record.getId(), response.getData(),
                    response.getExecutionTimeMs());
                return new AIInvokeResult(true, response.getData(), null, 
                    response.getExecutionTimeMs(), false, null);
            } else {
                return handleFallback(capability, scene, callerServiceId,
                    callerServiceName, requestPayload, 
                    response != null ? response.getError() : "AI service unavailable");
            }
        } catch (Exception e) {
            return handleFallback(capability, scene, callerServiceId,
                callerServiceName, requestPayload, e.getMessage());
        }
    }

    private AIInvokeResult handleFallback(AICapability capability, String scene,
            Long callerServiceId, String callerServiceName, String requestPayload,
            String reason) {
        
        if ("NONE".equals(capability.getFallbackType())) {
            domainService.markRequestFailed(
                domainService.createRequestRecord(
                    capability.getCapabilityId(), capability.getCapabilityName(),
                    scene, callerServiceId, callerServiceName, requestPayload
                ).getId(),
                reason
            );
            return new AIInvokeResult(false, null, reason, 0L, false, null);
        }
        
        AIRequest fakeRequest = new AIRequest(capability.getCapabilityId(), requestPayload);
        FallbackHandler.FallbackResponse fallbackResponse = 
            fallbackHandler.handleFallback(capability, fakeRequest);
        
        if (fallbackResponse.isSuccess()) {
            domainService.markRequestFallback(
                domainService.createRequestRecord(
                    capability.getCapabilityId(), capability.getCapabilityName(),
                    scene, callerServiceId, callerServiceName, requestPayload
                ).getId(),
                fallbackResponse.getFallbackType(),
                fallbackResponse.getReason(),
                fallbackResponse.getData(),
                0L
            );
            return new AIInvokeResult(true, fallbackResponse.getData(), null,
                0L, true, fallbackResponse.getReason());
        } else {
            domainService.markRequestFailed(
                domainService.createRequestRecord(
                    capability.getCapabilityId(), capability.getCapabilityName(),
                    scene, callerServiceId, callerServiceName, requestPayload
                ).getId(),
                reason
            );
            return new AIInvokeResult(false, null, reason, 0L, false, null);
        }
    }

    public static class AIInvokeResult {
        private final boolean success;
        private final String data;
        private final String error;
        private final long executionTimeMs;
        private final boolean usedFallback;
        private final String fallbackReason;

        public AIInvokeResult(boolean success, String data, String error,
                            long executionTimeMs, boolean usedFallback,
                            String fallbackReason) {
            this.success = success;
            this.data = data;
            this.error = error;
            this.executionTimeMs = executionTimeMs;
            this.usedFallback = usedFallback;
            this.fallbackReason = fallbackReason;
        }

        public boolean isSuccess() { return success; }
        public String getData() { return data; }
        public String getError() { return error; }
        public long getExecutionTimeMs() { return executionTimeMs; }
        public boolean isUsedFallback() { return usedFallback; }
        public String getFallbackReason() { return fallbackReason; }
        public Map<String, Object> toMap() {
            return Map.of(
                "success", success,
                "data", data != null ? data : "",
                "error", error != null ? error : "",
                "executionTimeMs", executionTimeMs,
                "usedFallback", usedFallback,
                "fallbackReason", fallbackReason != null ? fallbackReason : ""
            );
        }
    }
}