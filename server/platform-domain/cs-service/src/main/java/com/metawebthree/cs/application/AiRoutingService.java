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
        
        // Check if message contains tool invocation intent
        Optional<String> toolResult = tryInvokeTool(message);
        if (toolResult.isPresent()) {
            // Build context with tool result and get AI response
            List<Map<String, String>> messages = buildContext(sessionId, message, "查询结果: " + toolResult.get());
            return aiChatPort.chat(sessionId, messages);
        }
        
        // Normal chat flow
        List<Map<String, String>> messages = buildContext(sessionId, message, null);
        return aiChatPort.chat(sessionId, messages);
    }

    private Optional<String> tryInvokeTool(String userMessage) {
        String lowerMessage = userMessage.toLowerCase();
        
        // Check for order query intent
        if (containsAny(lowerMessage, "查订单", "订单查询", "我的订单", "订单状态", "order") && 
            containsAny(lowerMessage, "订单号", "订单编号", "orderid", "order_id")) {
            return invokeTool("QueryOrderTool", Map.of("orderId", extractOrderId(userMessage)));
        }
        
        // Check for logistics query intent
        if (containsAny(lowerMessage, "查物流", "物流查询", "快递查询", "物流信息", "logistics")) {
            return invokeTool("QueryLogisticsTool", Map.of("orderId", extractOrderId(userMessage)));
        }
        
        // Check for refund intent
        if (containsAny(lowerMessage, "申请退款", "退款", "退货退款", "refund")) {
            return invokeTool("InitiateRefundTool", Map.of("orderId", extractOrderId(userMessage), "reason", "用户申请退款"));
        }
        
        // Check for cancel order intent
        if (containsAny(lowerMessage, "取消订单", "取消", "撤销订单", "cancel")) {
            return invokeTool("CancelOrderTool", Map.of("orderId", extractOrderId(userMessage)));
        }
        
        return Optional.empty();
    }
    
    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (text.contains(keyword.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
    
    private String extractOrderId(String message) {
        // Try to extract order ID from message
        // Look for patterns like "订单号12345" or "orderId: 12345"
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(?:订单号|订单编号|order[_-]?id)[：:]?\\s*([A-Za-z0-9-]+)");
        java.util.regex.Matcher matcher = pattern.matcher(message);
        if (matcher.find()) {
            return matcher.group(1);
        }
        // Default order ID for testing
        return "ORDER_DEFAULT";
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
                    log.warn("Tool {} execution failed: {}", toolName, result.getResult());
                }
            }
        } catch (Exception e) {
            log.error("Error invoking tool {}: {}", toolName, e.getMessage());
        }
        return Optional.empty();
    }

    private List<Map<String, String>> buildContext(String sessionId, String userMessage, String toolResult) {
        List<Map<String, String>> messages = new ArrayList<>();
        
        // System message with tool descriptions
        Map<String, String> systemMessage = new HashMap<>();
        systemMessage.put("role", "system");
        String systemContent = buildSystemContent();
        systemMessage.put("content", systemContent);
        messages.add(systemMessage);
        
        // Load historical messages for multi-turn conversation
        List<Message> history = messageRepository.findBySessionId(sessionId);
        int maxHistory = 10; // Limit to last 10 messages
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
        
        // Current user message
        Map<String, String> userMsg = new HashMap<>();
        userMsg.put("role", "user");
        userMsg.put("content", userMessage);
        messages.add(userMsg);
        
        // If there's a tool result, add it as assistant message
        if (toolResult != null) {
            Map<String, String> toolMsg = new HashMap<>();
            toolMsg.put("role", "assistant");
            toolMsg.put("content", toolResult);
            messages.add(toolMsg);
        }
        
        return messages;
    }
    
    private String buildSystemContent() {
        StringBuilder sb = new StringBuilder();
        sb.append("你是电商客服助手，帮助用户解答商品、订单、售后等问题。回答简洁准确。\n");
        sb.append("你可以通过以下工具帮助用户：\n");
        
        for (AiTool tool : aiToolRegistry.getAllTools()) {
            sb.append("- ").append(tool.getName()).append(": ").append(tool.getDescription()).append("\n");
        }
        
        sb.append("当用户询问订单、物流、退款或取消订单时，请引导用户提供订单号。");
        return sb.toString();
    }
}
