package com.metawebthree.digitaltwin.infrastructure.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;


public abstract class AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(AbstractAIClient.class);
    private static final long RETRY_BACKOFF_BASE_MS = 100L;

    public static final ObjectMapper objectMapper = new ObjectMapper();

    protected final String endpoint;
    protected final int timeoutMs;
    protected final int retryCount;
    protected final HttpClient httpClient;

    protected AbstractAIClient(String endpoint, int timeoutMs, int retryCount) {
        this.endpoint = endpoint;
        this.timeoutMs = timeoutMs;
        this.retryCount = retryCount;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(timeoutMs))
            .build();
    }

    
    public AIClientResponse invoke(AIClientRequest request) {
        int attempts = 0;
        Exception lastException = null;

        while (attempts < retryCount) {
            try {
                AIClientResponse response = doInvoke(request);
                if (response.isSuccess()) {
                    return response;
                }
                log.warn("AI invocation attempt {} failed: {}", attempts + 1, response.getError());
            } catch (Exception e) {
                lastException = e;
                log.warn("AI invocation attempt {} threw exception: {}", attempts + 1, e.getMessage());
            }
            attempts++;
            if (attempts < retryCount) {
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

    
    protected abstract AIClientResponse doInvoke(AIClientRequest request);

    
    protected HttpRequest.Builder buildRequest(URI uri, String body) {
        return HttpRequest.newBuilder()
            .uri(uri)
            .timeout(Duration.ofMillis(timeoutMs))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body));
    }

    
    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/health"))
                .timeout(Duration.ofMillis(2000))
                .GET()
                .build();

            HttpResponse<String> response = httpClient.send(request,
                HttpResponse.BodyHandlers.ofString());
            return response.statusCode() >= 200 && response.statusCode() < 300;
        } catch (Exception e) {
            log.info("Health check failed for {}: {}", endpoint, e.getMessage());
            return false;
        }
    }

    public String getEndpoint() {
        return endpoint;
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