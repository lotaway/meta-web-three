package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Agent;
import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.repository.AgentRepository;
import com.metawebthree.cs.domain.repository.ConversationRepository;

import java.util.List;
import java.util.Optional;

public class AssignmentService {
    private final AgentRepository agentRepository;
    private final ConversationRepository conversationRepository;

    public AssignmentService(AgentRepository agentRepository,
                              ConversationRepository conversationRepository) {
        this.agentRepository = agentRepository;
        this.conversationRepository = conversationRepository;
    }

    public Optional<Agent> findAvailableAgent(Conversation conversation) {
        Long previousAgentId = conversation.getAgentId();
        if (previousAgentId != null) {
            Optional<Agent> previousAgent = agentRepository.findById(previousAgentId);
            if (previousAgent.isPresent() && previousAgent.get().isAvailable()) {
                return previousAgent;
            }
        }
        Long groupId = findAgentGroupId(conversation);
        if (groupId != null) {
            List<Agent> available = agentRepository.findAvailableByGroupId(groupId);
            if (!available.isEmpty()) {
                return Optional.of(available.get(0));
            }
        }
        List<Agent> allOnline = agentRepository.findAvailableByGroupId(null);
        return allOnline.stream()
                .filter(Agent::isAvailable)
                .min((a, b) -> Integer.compare(a.getCurrentLoad(), b.getCurrentLoad()));
    }

    private Long findAgentGroupId(Conversation conversation) {
        return null;
    }
}
