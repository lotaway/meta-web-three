package com.metawebthree.aiwarehouse.infrastructure.client;

import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class ForecastingClient extends AbstractAIClient {

    public ForecastingClient(String endpoint, int timeoutMs, int retryCount) {
        super(endpoint, timeoutMs, retryCount);
    }

    @Override
    protected AIResponse doInvoke(AIRequest request) {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/api/v1/forecast"))
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

class Duration {
    private final long millis;
    public Duration(long millis) { this.millis = millis; }
    public static Duration ofMillis(long millis) { return new Duration(millis); }
    public long toMillis() { return millis; }
}