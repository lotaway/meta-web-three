package com.metawebthree.aiwarehouse.infrastructure.client;

public interface AIClient {
    AIResponse invoke(AIRequest request);
    boolean isAvailable();
}

class AIRequest {
    private String capabilityId;
    private String payload;
    private Map<String, String> headers;

    public AIRequest(String capabilityId, String payload) {
        this.capabilityId = capabilityId;
        this.payload = payload;
        this.headers = new java.util.HashMap<>();
    }

    public String getCapabilityId() { return capabilityId; }
    public String getPayload() { return payload; }
    public Map<String, String> getHeaders() { return headers; }
    public void addHeader(String key, String value) { headers.put(key, value); }
}

class AIResponse {
    private boolean success;
    private String data;
    private String error;
    private long executionTimeMs;

    public static AIResponse success(String data, long executionTimeMs) {
        AIResponse response = new AIResponse();
        response.success = true;
        response.data = data;
        response.executionTimeMs = executionTimeMs;
        return response;
    }

    public static AIResponse failure(String error, long executionTimeMs) {
        AIResponse response = new AIResponse();
        response.success = false;
        response.error = error;
        response.executionTimeMs = executionTimeMs;
        return response;
    }

    public boolean isSuccess() { return success; }
    public String getData() { return data; }
    public String getError() { return error; }
    public long getExecutionTimeMs() { return executionTimeMs; }
}