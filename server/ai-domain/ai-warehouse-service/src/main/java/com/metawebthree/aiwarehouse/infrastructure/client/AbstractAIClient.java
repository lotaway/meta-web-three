package com.metawebthree.aiwarehouse.infrastructure.client;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public abstract class AbstractAIClient implements AIClient {
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

    protected AIResponse executeWithRetry(AIRequest request) {
        int attempts = 0;
        Exception lastException = null;

        while (attempts < retryCount) {
            try {
                attempts++;
                long startTime = System.currentTimeMillis();
                AIResponse response = doInvoke(request);
                response.executionTimeMs = System.currentTimeMillis() - startTime;
                return response;
            } catch (Exception e) {
                lastException = e;
                if (attempts < retryCount) {
                    try {
                        Thread.sleep(100L * attempts);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        return AIResponse.failure("Interrupted: " + ie.getMessage(), 0);
                    }
                }
            }
        }

        return AIResponse.failure(
            "Failed after " + retryCount + " attempts: " + 
            (lastException != null ? lastException.getMessage() : "Unknown error"), 
            0
        );
    }

    protected abstract AIResponse doInvoke(AIRequest request);

    @Override
    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/health"))
                .timeout(Duration.ofMillis(1000))
                .GET()
                .build();
            
            HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
            return response.statusCode() >= 200 && response.statusCode() < 300;
        } catch (Exception e) {
            return false;
        }
    }
}