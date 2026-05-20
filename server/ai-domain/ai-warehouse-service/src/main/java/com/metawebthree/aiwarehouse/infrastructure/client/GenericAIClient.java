package com.metawebthree.aiwarehouse.infrastructure.client;

import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class GenericAIClient extends AbstractAIClient {

    public GenericAIClient(String endpoint, int timeoutMs, int retryCount) {
        super(endpoint, timeoutMs, retryCount);
    }

    @Override
    protected AIResponse doInvoke(AIRequest request) {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(request.getPayload()))
                .build();

            HttpResponse<String> response = httpClient.send(httpRequest,
                HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() >= 200 && response.statusCode() < 300) {
                return AIResponse.success(response.body(), 0);
            } else {
                return AIResponse.failure("HTTP " + response.statusCode() + ": " + 
                    response.body(), 0);
            }
        } catch (Exception e) {
            return AIResponse.failure(e.getMessage(), 0);
        }
    }
}