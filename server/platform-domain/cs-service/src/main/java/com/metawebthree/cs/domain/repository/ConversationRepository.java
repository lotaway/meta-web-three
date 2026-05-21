package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.model.enums.ConversationStatus;

import java.util.List;
import java.util.Optional;

public interface ConversationRepository {
    Conversation save(Conversation conversation);
    Optional<Conversation> findBySessionId(String sessionId);
    Optional<Conversation> findActiveByCustomerId(Long customerId);
    List<Conversation> findByAgentId(Long agentId);
    List<Conversation> findByStatus(ConversationStatus status);
    List<Conversation> findByCustomerId(Long customerId);
}
