package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderSplitRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkOrderSplitRuleMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class WorkOrderSplitRuleRepositoryImpl implements WorkOrderSplitRuleRepository {
    private final WorkOrderSplitRuleMapper mapper;
    private final ObjectMapper json;

    public WorkOrderSplitRuleRepositoryImpl(WorkOrderSplitRuleMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public WorkOrderSplitRule save(WorkOrderSplitRule rule) {
        return persist(rule);
    }

    @Override
    public Optional<WorkOrderSplitRule> findById(Long id) {
        QueryWrapper<WorkOrderSplitRuleDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<WorkOrderSplitRule> findByRuleCode(String ruleCode) {
        QueryWrapper<WorkOrderSplitRuleDO> query = new QueryWrapper<>();
        query.eq("rule_code", ruleCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<WorkOrderSplitRule> findAll() {
        QueryWrapper<WorkOrderSplitRuleDO> query = new QueryWrapper<>();
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<WorkOrderSplitRule> findByEnabled(Boolean enabled) {
        QueryWrapper<WorkOrderSplitRuleDO> query = new QueryWrapper<>();
        query.eq("enabled", enabled);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<WorkOrderSplitRule> findBySplitType(String splitType) {
        QueryWrapper<WorkOrderSplitRuleDO> query = new QueryWrapper<>();
        query.eq("split_type", splitType);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private WorkOrderSplitRule persist(WorkOrderSplitRule entity) {
        WorkOrderSplitRuleDO dto = new WorkOrderSplitRuleDO();
        BeanUtils.copyProperties(entity, dto);
        try { dto.setConditions(json.writeValueAsString(entity.getConditions())); }
        catch (JsonProcessingException e) { throw new IllegalArgumentException("Invalid conditions JSON", e); }
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("WorkOrderSplitRule not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private WorkOrderSplitRule toEntity(WorkOrderSplitRuleDO dto) {
        WorkOrderSplitRule entity = new WorkOrderSplitRule();
        BeanUtils.copyProperties(dto, entity);
        try {
            entity.setConditions(dto.getConditions() == null ? null : json.readValue(dto.getConditions(), new TypeReference<List<WorkOrderSplitRule.SplitCondition>>() {}));
        } catch (JsonProcessingException e) { throw new IllegalStateException("Stored conditions JSON is invalid", e); }
        return entity;
    }
}