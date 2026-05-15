package com.metawebthree.cs.infrastructure.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.cs.domain.ports.AiChatPort;
import com.metawebthree.cs.dto.ChatCompletionRequest;
import com.metawebthree.cs.dto.ChatCompletionResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

public class LocalLlmProviderClient implements AiChatPort {
    private static final Logger log = LoggerFactory.getLogger(LocalLlmProviderClient.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    private final HttpClient httpClient;
    private final String baseUrl;
    private final String adminToken;

    public LocalLlmProviderClient(String baseUrl, String adminToken) {
        this.baseUrl = baseUrl;
        this.adminToken = adminToken;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    @Override
    public String chat(String sessionId, List<Map<String, String>> messages) {
        try {
            ChatCompletionRequest request = new ChatCompletionRequest("default", messages);
            String body = objectMapper.writeValueAsString(request);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + adminToken)
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> response = httpClient.send(httpRequest,
                    HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                log.warn("ai provider returned status:{} session:{}", response.statusCode(), sessionId);
                return "";
            }
            ChatCompletionResponse chatResponse = objectMapper.readValue(
                    response.body(), ChatCompletionResponse.class);
            return chatResponse.extractContent();
        } catch (Exception e) {
            log.error("ai chat failed session:{}", sessionId, e);
            return "";
        }
    }

    @Override
    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/"))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }
}
