package com.metawebthree.cs.application;

import com.metawebthree.cs.ai.tools.AiTool;
import com.metawebthree.cs.ai.tools.AiToolRegistry;
import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.model.Message;
import com.metawebthree.cs.domain.model.enums.SenderType;
import com.metawebthree.cs.domain.ports.AiChatPort;
import com.metawebthree.cs.domain.repository.MessageRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.function.Function;

@Service
public class AiRoutingService {
    private static final Logger log = LoggerFactory.getLogger(AiRoutingService.class);

    private final AiChatPort aiChatPort;
    private final MessageRepository messageRepository;
    private final AiToolRegistry aiToolRegistry;

    public AiRoutingService(AiChatPort aiChatPort, MessageRepository messageRepository, AiToolRegistry aiToolRegistry) {
        this.aiChatPort = aiChatPort;
        this.messageRepository = messageRepository;
        this.aiToolRegistry = aiToolRegistry;
    }

    public String processWithAi(String sessionId, Long customerId, String message) {
        if (!aiChatPort.isAvailable()) {
            return "";
        }

        List<Map<String, String>> messages = buildContext(sessionId, message);
        List<Map<String, Object>> toolDefs = buildToolDefinitions();

        if (toolDefs.isEmpty()) {
            return aiChatPort.chat(sessionId, messages);
        }

        Function<Map<String, Object>, String> toolExecutor = params -> {
            String toolName = (String) params.get("toolName");
            if (toolName == null) return "error: missing toolName";
            Map<String, Object> toolParams = (Map<String, Object>) params.get("params");
            if (toolParams == null) toolParams = new HashMap<>();
            Optional<String> result = invokeTool(toolName, toolParams);
            return result.orElse("tool execution failed");
        };

        return aiChatPort.chatWithTools(sessionId, messages, toolDefs, toolExecutor);
    }

    private List<Map<String, Object>> buildToolDefinitions() {
        List<Map<String, Object>> defs = new ArrayList<>();
        for (AiTool tool : aiToolRegistry.getAllTools()) {
            Map<String, Object> functionDef = new LinkedHashMap<>();
            functionDef.put("type", "function");
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", tool.getName());
            function.put("description", tool.getDescription());
            function.put("parameters", tool.getParameterSchema());
            functionDef.put("function", function);
            defs.add(functionDef);
        }
        return defs;
    }

    private Optional<String> invokeTool(String toolName, Map<String, Object> params) {
        try {
            Optional<AiTool> toolOpt = aiToolRegistry.getTool(toolName);
            if (toolOpt.isPresent()) {
                AiTool tool = toolOpt.get();
                com.metawebthree.cs.dto.AiToolRequest request = new com.metawebthree.cs.dto.AiToolRequest();
                request.setParams(params);
                com.metawebthree.cs.dto.AiToolResult result = tool.execute(request);
                if (result.isSuccess()) {
                    return Optional.of(String.valueOf(result.getResult()));
                } else {
                    log.warn("tool {} execution failed: {}", toolName, result.getResult());
                }
            }
        } catch (Exception e) {
            log.error("error invoking tool {}: {}", toolName, e.getMessage());
        }
        return Optional.empty();
    }

    private List<Map<String, String>> buildContext(String sessionId, String userMessage) {
        List<Map<String, String>> messages = new ArrayList<>();

        Map<String, String> systemMessage = new HashMap<>();
        systemMessage.put("role", "system");
        systemMessage.put("content", "你是电商客服助手，回答简洁准确。如果需要查询订单、物流、退款或取消订单，请使用提供的工具。");
        messages.add(systemMessage);

        List<Message> history = messageRepository.findBySessionId(sessionId);
        int maxHistory = 10;
        int start = Math.max(0, history.size() - maxHistory);
        for (int i = start; i < history.size(); i++) {
            Message msg = history.get(i);
            Map<String, String> msgMap = new HashMap<>();
            if (msg.getSenderType() == SenderType.CUSTOMER) {
                msgMap.put("role", "user");
            } else if (msg.getSenderType() == SenderType.AGENT || msg.getSenderType() == SenderType.SYSTEM) {
                msgMap.put("role", "assistant");
            } else {
                continue;
            }
            msgMap.put("content", msg.getContent() != null ? msg.getContent() : "");
            messages.add(msgMap);
        }

        Map<String, String> userMsg = new HashMap<>();
        userMsg.put("role", "user");
        userMsg.put("content", userMessage);
        messages.add(userMsg);

        return messages;
    }
}
