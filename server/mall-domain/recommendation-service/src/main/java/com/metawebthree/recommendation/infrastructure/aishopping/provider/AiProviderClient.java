package com.metawebthree.recommendation.infrastructure.aishopping.provider;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.recommendation.application.aishopping.AiProviderConfig;
import com.metawebthree.recommendation.application.aishopping.AiProviderSettings;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;

/**
 * AI provider client speaking the OpenAI-compatible protocol (text embedding /
 * image embedding / LLM chat). Configuration comes from AiProviderConfig
 * (application.yml plus DB overrides).
 */
@Component
public class AiProviderClient {

    private static final String DEFAULT_TEXT_EMBEDDING_PATH = "/v1/embeddings";
    private static final String DEFAULT_IMAGE_EMBEDDING_PATH = "/v1/images/embeddings";
    private static final String DEFAULT_CHAT_PATH = "/v1/chat/completions";

    private final AiProviderConfig config;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    public AiProviderClient(AiProviderConfig config) {
        this.config = config;
    }

    public float[] embedText(String text) {
        AiProviderSettings.Endpoint endpoint = config.getSettings().getEmbedding();
        if (!endpoint.isConfigured()) {
            throw new IllegalStateException("text embedding provider is not configured");
        }
        String url = joinUrl(endpoint.baseUrl, endpoint.resolvePath(DEFAULT_TEXT_EMBEDDING_PATH));
        Object body = Map.of(
                "model", endpoint.model,
                "input", List.of(text));
        String responseBody = post(url, endpoint, body);
        return parseEmbedding(responseBody);
    }

    public float[] embedImage(byte[] imageBytes) {
        AiProviderSettings.Endpoint endpoint = config.getSettings().getImageEmbedding();
        if (!endpoint.isConfigured()) {
            throw new IllegalStateException("image embedding provider is not configured");
        }
        String dataUrl = "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(imageBytes);
        String url = joinUrl(endpoint.baseUrl, endpoint.resolvePath(DEFAULT_IMAGE_EMBEDDING_PATH));
        Object body = Map.of(
                "model", endpoint.model,
                "input", List.of(Map.of("image_url", dataUrl)));
        String responseBody = post(url, endpoint, body);
        return parseEmbedding(responseBody);
    }

    /** Runs an LLM chat and returns the plain-text reply. For structured output the caller provides the prompt and requires JSON. */
    public String chat(String systemPrompt, String userPrompt) {
        AiProviderSettings.Endpoint endpoint = config.getSettings().getLlm();
        if (!endpoint.isConfigured()) {
            throw new IllegalStateException("llm provider is not configured");
        }
        String url = joinUrl(endpoint.baseUrl, endpoint.resolvePath(DEFAULT_CHAT_PATH));
        Object body = Map.of(
                "model", endpoint.model,
                "temperature", 0,
                "messages", List.of(
                        Map.of("role", "system", "content", systemPrompt),
                        Map.of("role", "user", "content", userPrompt)));
        String responseBody = post(url, endpoint, body);
        try {
            JsonNode root = objectMapper.readTree(responseBody);
            JsonNode content = root.path("choices").path(0).path("message").path("content");
            if (content.isMissingNode() || content.isNull()) {
                throw new IllegalStateException("unexpected llm response: " + responseBody);
            }
            return content.asText();
        } catch (Exception e) {
            throw new IllegalStateException("failed to parse llm response: " + e.getMessage(), e);
        }
    }

    private float[] parseEmbedding(String responseBody) {
        try {
            JsonNode root = objectMapper.readTree(responseBody);
            JsonNode data = root.path("data");
            if (!data.isArray() || data.isEmpty()) {
                throw new IllegalStateException("embedding response has no data: " + responseBody);
            }
            JsonNode vector = data.get(0).path("embedding");
            List<Float> values = new ArrayList<>();
            vector.forEach(v -> values.add((float) v.asDouble()));
            float[] result = new float[values.size()];
            for (int i = 0; i < values.size(); i++) {
                result[i] = values.get(i);
            }
            if (result.length == 0) {
                throw new IllegalStateException("embedding vector is empty");
            }
            return result;
        } catch (Exception e) {
            throw new IllegalStateException("failed to parse embedding response: " + e.getMessage(), e);
        }
    }

    private String post(String url, AiProviderSettings.Endpoint endpoint, Object body) {
        Exception lastError = null;
        for (int attempt = 0; attempt <= endpoint.maxRetries; attempt++) {
            try {
                return sendOnce(url, endpoint, body);
            } catch (Exception e) {
                lastError = e;
                if (attempt < endpoint.maxRetries && !backoff(attempt)) {
                    break;
                }
            }
        }
        throw new IllegalStateException("ai provider request failed: "
                + (lastError != null ? lastError.getMessage() : "unknown error"), lastError);
    }

    private String sendOnce(String url, AiProviderSettings.Endpoint endpoint, Object body) throws Exception {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMillis(endpoint.timeoutMs))
                .header("Content-Type", "application/json");
        if (endpoint.apiKey != null && !endpoint.apiKey.isBlank()) {
            builder.header("Authorization", "Bearer " + endpoint.apiKey);
        }
        builder.POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)));
        HttpResponse<String> response = httpClient.send(builder.build(),
                HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() >= 200 && response.statusCode() < 300) {
            return response.body();
        }
        throw new IllegalStateException("HTTP " + response.statusCode() + ": " + response.body());
    }

    private boolean backoff(int attempt) {
        try {
            Thread.sleep(100L * (attempt + 1));
            return true;
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private String joinUrl(String baseUrl, String path) {
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        String p = path.startsWith("/") ? path : "/" + path;
        return base + p;
    }
}
