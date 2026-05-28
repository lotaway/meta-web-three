package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.QcTriggerRule;
import com.metawebthree.mes.domain.entity.QcTriggerRule.TriggerCondition;
import com.metawebthree.mes.domain.entity.QcTriggerRule.TriggerType;
import com.metawebthree.mes.domain.repository.QcTriggerRuleRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.QcTriggerRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.QcTriggerRuleMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class QcTriggerRuleRepositoryImpl implements QcTriggerRuleRepository {
    
    @Autowired
    private QcTriggerRuleMapper qcTriggerRuleMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<QcTriggerRule> findById(Long id) {
        QcTriggerRuleDO triggerRuleDO = qcTriggerRuleMapper.selectById(id);
        return Optional.ofNullable(triggerRuleDO).map(this::toEntity);
    }
    
    @Override
    public Optional<QcTriggerRule> findByRuleCode(String ruleCode) {
        LambdaQueryWrapper<QcTriggerRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcTriggerRuleDO::getRuleCode, ruleCode);
        QcTriggerRuleDO triggerRuleDO = qcTriggerRuleMapper.selectOne(wrapper);
        return Optional.ofNullable(triggerRuleDO).map(this::toEntity);
    }
    
    @Override
    public List<QcTriggerRule> findAll() {
        List<QcTriggerRuleDO> doList = qcTriggerRuleMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<QcTriggerRule> findByTriggerType(TriggerType triggerType) {
        LambdaQueryWrapper<QcTriggerRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcTriggerRuleDO::getTriggerType, triggerType.name());
        List<QcTriggerRuleDO> doList = qcTriggerRuleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<QcTriggerRule> findByIsEnabled(Boolean isEnabled) {
        LambdaQueryWrapper<QcTriggerRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcTriggerRuleDO::getIsEnabled, isEnabled);
        List<QcTriggerRuleDO> doList = qcTriggerRuleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public QcTriggerRule save(QcTriggerRule triggerRule) {
        QcTriggerRuleDO triggerRuleDO = toDO(triggerRule);
        if (triggerRule.getId() == null) {
            qcTriggerRuleMapper.insert(triggerRuleDO);
            triggerRule.setId(triggerRuleDO.getId());
        } else {
            qcTriggerRuleMapper.updateById(triggerRuleDO);
        }
        return triggerRule;
    }
    
    @Override
    public void update(QcTriggerRule triggerRule) {
        if (triggerRule.getId() != null) {
            QcTriggerRuleDO triggerRuleDO = toDO(triggerRule);
            qcTriggerRuleMapper.updateById(triggerRuleDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        qcTriggerRuleMapper.deleteById(id);
    }
    
    @Override
    public Boolean existsByRuleCode(String ruleCode) {
        LambdaQueryWrapper<QcTriggerRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcTriggerRuleDO::getRuleCode, ruleCode);
        return qcTriggerRuleMapper.selectCount(wrapper) > 0;
    }
    
    private QcTriggerRule toEntity(QcTriggerRuleDO doObj) {
        if (doObj == null) {
            return null;
        }
        QcTriggerRule entity = new QcTriggerRule();
        entity.setId(doObj.getId());
        entity.setRuleCode(doObj.getRuleCode());
        entity.setRuleName(doObj.getRuleName());
        if (doObj.getTriggerType() != null) {
            entity.setTriggerType(TriggerType.valueOf(doObj.getTriggerType()));
        }
        entity.setTargetObject(doObj.getTargetObject());
        if (doObj.getConditionJson() != null) {
            try {
                TriggerCondition condition = objectMapper.readValue(doObj.getConditionJson(), TriggerCondition.class);
                entity.setCondition(condition);
            } catch (JsonProcessingException e) {
                entity.setCondition(new TriggerCondition());
            }
        }
        entity.setInspectionType(doObj.getInspectionType());
        entity.setInspectionPlanCode(doObj.getInspectionPlanCode());
        entity.setIsEnabled(doObj.getIsEnabled());
        entity.setPriority(doObj.getPriority());
        return entity;
    }
    
    private QcTriggerRuleDO toDO(QcTriggerRule entity) {
        if (entity == null) {
            return null;
        }
        QcTriggerRuleDO doObj = new QcTriggerRuleDO();
        doObj.setId(entity.getId());
        doObj.setRuleCode(entity.getRuleCode());
        doObj.setRuleName(entity.getRuleName());
        doObj.setTriggerType(entity.getTriggerType() != null ? entity.getTriggerType().name() : null);
        doObj.setTargetObject(entity.getTargetObject());
        if (entity.getCondition() != null) {
            try {
                doObj.setConditionJson(objectMapper.writeValueAsString(entity.getCondition()));
            } catch (JsonProcessingException e) {
                doObj.setConditionJson("{}");
            }
        }
        doObj.setInspectionType(entity.getInspectionType());
        doObj.setInspectionPlanCode(entity.getInspectionPlanCode());
        doObj.setIsEnabled(entity.getIsEnabled());
        doObj.setPriority(entity.getPriority());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}