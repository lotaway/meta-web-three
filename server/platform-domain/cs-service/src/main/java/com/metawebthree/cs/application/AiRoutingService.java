package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.ports.AiChatPort;
import com.metawebthree.cs.domain.repository.MessageRepository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class AiRoutingService {
    private final AiChatPort aiChatPort;
    private final MessageRepository messageRepository;

    public AiRoutingService(AiChatPort aiChatPort, MessageRepository messageRepository) {
        this.aiChatPort = aiChatPort;
        this.messageRepository = messageRepository;
    }

    public String processWithAi(String sessionId, Long customerId, String message) {
        if (!aiChatPort.isAvailable()) {
            return "";
        }
        List<Map<String, String>> messages = buildContext(sessionId, message);
        return aiChatPort.chat(sessionId, messages);
    }

    private List<Map<String, String>> buildContext(String sessionId, String userMessage) {
        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> systemMessage = new HashMap<>();
        systemMessage.put("role", "system");
        systemMessage.put("content", "你是电商客服助手，帮助用户解答商品、订单、售后等问题。回答简洁准确。");
        messages.add(systemMessage);
        Map<String, String> userMsg = new HashMap<>();
        userMsg.put("role", "user");
        userMsg.put("content", userMessage);
        messages.add(userMsg);
        return messages;
    }
}
