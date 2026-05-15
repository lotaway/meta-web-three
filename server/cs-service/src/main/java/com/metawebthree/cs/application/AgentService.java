package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Agent;
import com.metawebthree.cs.domain.repository.AgentRepository;

import java.util.List;
import java.util.Optional;

public class AgentService {
    private final AgentRepository agentRepository;

    public AgentService(AgentRepository agentRepository) {
        this.agentRepository = agentRepository;
    }

    public Agent create(Long adminId, String nickname, Long groupId) {
        Agent agent = new Agent();
        agent.setAdminId(adminId);
        agent.setNickname(nickname);
        agent.setGroupId(groupId);
        agent.setStatus("OFFLINE");
        agent.setMaxConcurrent(5);
        agent.setCurrentLoad(0);
        return agentRepository.save(agent);
    }

    public void goOnline(Long id) {
        agentRepository.updateStatus(id, "ONLINE");
    }

    public void goOffline(Long id) {
        agentRepository.updateStatus(id, "OFFLINE");
    }

    public void setBusy(Long id) {
        agentRepository.updateStatus(id, "BUSY");
    }

    public void incrementLoad(Long id) {
        agentRepository.updateLoad(id, 1);
    }

    public void decrementLoad(Long id) {
        agentRepository.updateLoad(id, -1);
    }

    public Optional<Agent> findById(Long id) {
        return agentRepository.findById(id);
    }

    public List<Agent> listOnline() {
        return agentRepository.findByStatus("ONLINE");
    }

    public List<Agent> listByGroup(Long groupId) {
        return agentRepository.findByGroupId(groupId);
    }
}
