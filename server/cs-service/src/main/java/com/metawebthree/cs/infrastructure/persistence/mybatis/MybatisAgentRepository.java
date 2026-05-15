package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.cs.domain.model.Agent;
import com.metawebthree.cs.domain.repository.AgentRepository;

import java.util.List;
import java.util.Optional;

public class MybatisAgentRepository implements AgentRepository {
    private final MybatisAgentMapper mapper;

    public MybatisAgentRepository(MybatisAgentMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Agent save(Agent agent) {
        if (agent.getId() == null) {
            mapper.insert(agent);
        } else {
            mapper.updateById(agent);
        }
        return agent;
    }

    @Override
    public Optional<Agent> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public Optional<Agent> findByAdminId(Long adminId) {
        LambdaQueryWrapper<Agent> query = new LambdaQueryWrapper<Agent>()
                .eq(Agent::getAdminId, adminId);
        return Optional.ofNullable(mapper.selectOne(query));
    }

    @Override
    public List<Agent> findByStatus(String status) {
        LambdaQueryWrapper<Agent> query = new LambdaQueryWrapper<Agent>()
                .eq(Agent::getStatus, status);
        return mapper.selectList(query);
    }

    @Override
    public List<Agent> findByGroupId(Long groupId) {
        LambdaQueryWrapper<Agent> query = new LambdaQueryWrapper<Agent>()
                .eq(Agent::getGroupId, groupId);
        return mapper.selectList(query);
    }

    @Override
    public List<Agent> findAvailableByGroupId(Long groupId) {
        LambdaQueryWrapper<Agent> query = new LambdaQueryWrapper<Agent>()
                .eq(Agent::getStatus, "ONLINE")
                .apply("current_load < max_concurrent");
        if (groupId != null) {
            query.eq(Agent::getGroupId, groupId);
        }
        return mapper.selectList(query);
    }

    @Override
    public void updateStatus(Long id, String status) {
        mapper.updateStatus(id, status);
    }

    @Override
    public void updateLoad(Long id, int delta) {
        mapper.updateLoad(id, delta);
    }
}
