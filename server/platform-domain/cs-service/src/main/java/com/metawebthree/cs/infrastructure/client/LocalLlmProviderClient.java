package com.metawebthree.cs.infrastructure.client;

import com.fasterxml.jackson.core.type.TypeReference;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class LocalLlmProviderClient implements AiChatPort {
    private static final Logger log = LoggerFactory.getLogger(LocalLlmProviderClient.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final int MAX_TOOL_ROUNDS = 5;

    private final HttpClient httpClient;
    private final String baseUrl;
    private final String adminToken;
    private final String model;

    public LocalLlmProviderClient(String baseUrl, String adminToken, String model) {
        this.baseUrl = baseUrl;
        this.adminToken = adminToken;
        this.model = model != null ? model : "default";
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    public LocalLlmProviderClient(String baseUrl, String adminToken) {
        this(baseUrl, adminToken, "default");
    }

    @Override
    public String chat(String sessionId, List<Map<String, String>> messages) {
        try {
            List<Map<String, Object>> extendedMessages = new ArrayList<>();
            for (Map<String, String> msg : messages) {
                extendedMessages.add(new HashMap<>(msg));
            }
            ChatCompletionRequest request = new ChatCompletionRequest(model, extendedMessages);
            String body = objectMapper.writeValueAsString(request);
            ChatCompletionResponse response = doHttpCall(body, sessionId);
            if (response == null) return "";
            return response.extractContent();
        } catch (Exception e) {
            log.error("ai chat failed session:{}", sessionId, e);
            return "";
        }
    }

    @Override
    public String chatWithTools(String sessionId, List<Map<String, String>> messages,
                                List<Map<String, Object>> tools,
                                Function<Map<String, Object>, String> toolExecutor) {
        try {
            List<Map<String, Object>> extendedMessages = new ArrayList<>();
            for (Map<String, String> msg : messages) {
                extendedMessages.add(new HashMap<>(msg));
            }

            ChatCompletionRequest request = new ChatCompletionRequest(model, extendedMessages);
            request.setTools(tools);

            for (int round = 0; round < MAX_TOOL_ROUNDS; round++) {
                String body = objectMapper.writeValueAsString(request);
                ChatCompletionResponse response = doHttpCall(body, sessionId);
                if (response == null) return "";

                if (!response.hasToolCalls()) {
                    return response.extractContent();
                }

                List<Map<String, Object>> toolCalls = response.getToolCalls();
                Map<String, Object> assistantMsg = new HashMap<>();
                assistantMsg.put("role", "assistant");
                assistantMsg.put("content", null);
                assistantMsg.put("tool_calls", toolCalls);
                request.getMessages().add(assistantMsg);

                for (Map<String, Object> toolCall : toolCalls) {
                    String toolCallId = (String) toolCall.get("id");
                    Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
                    String toolName = (String) function.get("name");
                    String argumentsStr = (String) function.get("arguments");

                    Map<String, Object> arguments;
                    try {
                        arguments = objectMapper.readValue(argumentsStr,
                                new TypeReference<Map<String, Object>>() {});
                    } catch (Exception e) {
                        arguments = Map.of();
                    }

                    String toolResult = toolExecutor.apply(arguments);

                    Map<String, Object> toolMsg = new HashMap<>();
                    toolMsg.put("role", "tool");
                    toolMsg.put("tool_call_id", toolCallId);
                    toolMsg.put("content", toolResult);
                    request.getMessages().add(toolMsg);
                }

                request.setToolChoice(null);
            }

            log.warn("exceeded max tool rounds {} session:{}", MAX_TOOL_ROUNDS, sessionId);
            return "";
        } catch (Exception e) {
            log.error("ai chat with tools failed session:{}", sessionId, e);
            return "";
        }
    }

    private ChatCompletionResponse doHttpCall(String body, String sessionId) {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + adminToken)
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> httpResponse = httpClient.send(httpRequest,
                    HttpResponse.BodyHandlers.ofString());
            if (httpResponse.statusCode() != 200) {
                log.warn("ai provider returned status:{} session:{}", httpResponse.statusCode(), sessionId);
                return null;
            }
            return objectMapper.readValue(httpResponse.body(), ChatCompletionResponse.class);
        } catch (Exception e) {
            log.error("ai http call failed session:{}", sessionId, e);
            return null;
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
