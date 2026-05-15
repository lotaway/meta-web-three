package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.Agent;

import java.util.List;
import java.util.Optional;

public interface AgentRepository {
    Agent save(Agent agent);
    Optional<Agent> findById(Long id);
    Optional<Agent> findByAdminId(Long adminId);
    List<Agent> findByStatus(String status);
    List<Agent> findByGroupId(Long groupId);
    List<Agent> findAvailableByGroupId(Long groupId);
    void updateStatus(Long id, String status);
    void updateLoad(Long id, int delta);
    void deleteById(Long id);
}
