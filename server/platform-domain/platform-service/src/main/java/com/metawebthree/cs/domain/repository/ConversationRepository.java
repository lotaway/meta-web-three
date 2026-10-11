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

    // Paged queries
    List<Conversation> findByAgentId(Long agentId, int offset, int limit);
    long countByAgentId(Long agentId);
    List<Conversation> findByCustomerId(Long customerId, int offset, int limit);
    long countByCustomerId(Long customerId);
    List<Conversation> findByStatus(ConversationStatus status, int offset, int limit);
    long countByStatus(ConversationStatus status);
}
