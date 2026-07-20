package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.OpportunityMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class OpportunityRepositoryImpl implements OpportunityRepository {

    private final OpportunityMapper mapper;

    public OpportunityRepositoryImpl(OpportunityMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Opportunity> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<Opportunity> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public List<Opportunity> findByStage(String stage) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Opportunity>()
                .eq(Opportunity::getStage, stage));
    }

    @Override
    public List<Opportunity> findByPipelineId(Long pipelineId) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Opportunity>()
                .eq(Opportunity::getPipelineId, pipelineId));
    }

    @Override
    public List<Opportunity> findByAssignedTo(String assignedTo) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Opportunity>()
                .eq(Opportunity::getAssignedTo, assignedTo));
    }

    @Override
    public List<Opportunity> findByCustomerId(Long customerId) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Opportunity>()
                .eq(Opportunity::getCustomerId, customerId));
    }

    @Override
    public List<Opportunity> searchByKeyword(String keyword) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Opportunity>()
                .and(w -> w.like(Opportunity::getTitle, keyword).or().like(Opportunity::getOpportunityNo, keyword)
                        .or().like(Opportunity::getCompetitor, keyword))
                .orderByAsc(Opportunity::getOpportunityNo));
    }

    @Override
    public Opportunity insert(Opportunity opportunity) {
        mapper.insert(opportunity);
        return opportunity;
    }

    @Override
    public Opportunity updateById(Opportunity opportunity) {
        mapper.updateById(opportunity);
        return opportunity;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
