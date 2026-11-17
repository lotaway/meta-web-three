package com.metawebthree.digitaltwin.infrastructure.client;

import java.util.Map;

import org.apache.dubbo.config.annotation.DubboReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.generated.rpc.AICapabilityRequest;
import com.metawebthree.common.generated.rpc.AICapabilityResponse;
import com.metawebthree.common.generated.rpc.AICapabilityService;

/**
 * Base for factory AI clients. AI capabilities are reached through the
 * existing ai-service over Dubbo (service-to-service call rule); no caller
 * embeds an HTTP endpoint any more. The concrete AI/LLM runtimes behind
 * ai-service are external systems and stay HTTP there.
 */
public abstract class AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(AbstractAIClient.class);
    private static final long RETRY_BACKOFF_BASE_MS = 100L;

    public static final ObjectMapper objectMapper = new ObjectMapper();

    @DubboReference(check = false, lazy = true)
    protected AICapabilityService aiCapabilityService;

    protected AbstractAIClient() {
    }

    /**
     * Invoke an AI capability via Dubbo with the existing retry policy.
     */
    public AIClientResponse invoke(AIClientRequest request) {
        int attempts = 0;
        Exception lastException = null;

        while (attempts < 3) {
            try {
                AICapabilityResponse response = aiCapabilityService.invoke(
                        AICapabilityRequest.newBuilder()
                                .setCapability(request.getCapability())
                                .setPayload(request.toJson())
                                .build());
                if (response.getSuccess()) {
                    return AIClientResponse.success(response.getData(), 200);
                }
                log.warn("AI invocation attempt {} failed: {}", attempts + 1, response.getError());
            } catch (Exception e) {
                lastException = e;
                log.warn("AI invocation attempt {} threw exception: {}", attempts + 1, e.getMessage());
            }
            attempts++;
            if (attempts < 3) {
                try {
                    Thread.sleep(RETRY_BACKOFF_BASE_MS * attempts);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return AIClientResponse.failure("Interrupted during retry", -1);
                }
            }
        }

        return AIClientResponse.failure(
            lastException != null ? lastException.getMessage() : "Max retries exceeded",
            -1
        );
    }

    public static class AIClientRequest {
        private final String capability;
        private final Map<String, Object> payload;

        public AIClientRequest(String capability, Map<String, Object> payload) {
            this.capability = capability;
            this.payload = payload;
        }

        public String getCapability() {
            return capability;
        }

        public Map<String, Object> getPayload() {
            return payload;
        }

        public String toJson() {
            try {
                return objectMapper.writeValueAsString(payload);
            } catch (Exception e) {
                return "{}";
            }
        }
    }

    public static class AIClientResponse {
        private final boolean success;
        private final String data;
        private final String error;
        private final int statusCode;

        private AIClientResponse(boolean success, String data, String error, int statusCode) {
            this.success = success;
            this.data = data;
            this.error = error;
            this.statusCode = statusCode;
        }

        public static AIClientResponse success(String data, int statusCode) {
            return new AIClientResponse(true, data, null, statusCode);
        }

        public static AIClientResponse failure(String error, int statusCode) {
            return new AIClientResponse(false, null, error, statusCode);
        }

        public boolean isSuccess() {
            return success;
        }

        public String getData() {
            return data;
        }

        public String getError() {
            return error;
        }

        public int getStatusCode() {
            return statusCode;
        }

        @SuppressWarnings("unchecked")
        public Map<String, Object> getDataAsMap() {
            if (data == null) {
                return Map.of();
            }
            try {
                return objectMapper.readValue(data, Map.class);
            } catch (Exception e) {
                return Map.of();
            }
        }
    }
}
