package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.model.enums.ChannelType;
import com.metawebthree.cs.domain.model.enums.ConversationEvent;
import com.metawebthree.cs.domain.model.enums.ConversationStatus;
import com.metawebthree.cs.domain.ports.ConversationEventPort;
import com.metawebthree.cs.domain.repository.ConversationRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public class ConversationService {
    private final ConversationRepository conversationRepository;
    private final ConversationEventPort eventPort;

    public ConversationService(ConversationRepository conversationRepository,
                                ConversationEventPort eventPort) {
        this.conversationRepository = conversationRepository;
        this.eventPort = eventPort;
    }

    public Conversation create(Long customerId, ChannelType channel,
                                Long productId, Long orderId) {
        Conversation conversation = new Conversation(UUID.randomUUID().toString(), customerId, channel);
        conversation.setProductId(productId);
        conversation.setOrderId(orderId);
        Conversation saved = conversationRepository.save(conversation);
        eventPort.publish(saved.getSessionId(), ConversationEvent.CREATED, "customer:" + customerId);
        return saved;
    }

    public void assign(String sessionId, Long agentId) {
        Conversation conversation = conversationRepository.findBySessionId(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("conversation not found: " + sessionId));
        conversation.setAgentId(agentId);
        conversation.setStatus(ConversationStatus.ACTIVE);
        conversation.setActiveTime(LocalDateTime.now());
        conversationRepository.save(conversation);
        eventPort.publish(sessionId, ConversationEvent.ASSIGNED, "agent:" + agentId);
    }

    public void close(String sessionId) {
        Conversation conversation = conversationRepository.findBySessionId(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("conversation not found: " + sessionId));
        conversation.setStatus(ConversationStatus.CLOSED);
        conversation.setEndTime(LocalDateTime.now());
        conversationRepository.save(conversation);
        eventPort.publish(sessionId, ConversationEvent.CLOSED, "closed");
    }

    public void rate(String sessionId, Integer score) {
        Conversation conversation = conversationRepository.findBySessionId(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("conversation not found: " + sessionId));
        conversation.setSatisfactionScore(score);
        conversationRepository.save(conversation);
        eventPort.publish(sessionId, ConversationEvent.RATED, "score:" + score);
    }

    public Optional<Conversation> findActiveByCustomer(Long customerId) {
        return conversationRepository.findActiveByCustomerId(customerId);
    }

    public List<Conversation> listByCustomer(Long customerId) {
        return conversationRepository.findByCustomerId(customerId);
    }

    public List<Conversation> listByAgent(Long agentId) {
        return conversationRepository.findByAgentId(agentId);
    }

    public List<Conversation> listQueuing() {
        return conversationRepository.findByStatus(ConversationStatus.QUEUING);
    }
}
