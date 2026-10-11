package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.repository.LeadRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.LeadMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class LeadRepositoryImpl implements LeadRepository {

    private final LeadMapper mapper;

    public LeadRepositoryImpl(LeadMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Lead> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<Lead> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public List<Lead> findByStatus(String status) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Lead>()
                .eq(Lead::getStatus, status).orderByAsc(Lead::getLeadNo));
    }

    @Override
    public List<Lead> findBySource(String source) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Lead>()
                .eq(Lead::getSource, source));
    }

    @Override
    public List<Lead> findByAssignedTo(String assignedTo) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Lead>()
                .eq(Lead::getAssignedTo, assignedTo));
    }

    @Override
    public List<Lead> searchByKeyword(String keyword) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Lead>()
                .and(w -> w.like(Lead::getName, keyword).or().like(Lead::getCompany, keyword)
                        .or().like(Lead::getEmail, keyword).or().like(Lead::getPhone, keyword)
                        .or().like(Lead::getMobile, keyword))
                .orderByAsc(Lead::getLeadNo));
    }

    @Override
    public Lead insert(Lead lead) {
        mapper.insert(lead);
        return lead;
    }

    @Override
    public Lead updateById(Lead lead) {
        mapper.updateById(lead);
        return lead;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
