package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.CodeRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.CodeRuleMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
public class CodeRuleRepositoryImpl implements CodeRuleRepository {
    
    @Autowired
    private CodeRuleMapper codeRuleMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<CodeRule> findById(Long id) {
        CodeRuleDO codeRuleDO = codeRuleMapper.selectById(id);
        return Optional.ofNullable(codeRuleDO).map(this::toEntity);
    }
    
    @Override
    public Optional<CodeRule> findByRuleCode(String ruleCode) {
        LambdaQueryWrapper<CodeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CodeRuleDO::getRuleCode, ruleCode);
        CodeRuleDO codeRuleDO = codeRuleMapper.selectOne(wrapper);
        return Optional.ofNullable(codeRuleDO).map(this::toEntity);
    }
    
    @Override
    public Optional<CodeRule> findByBusinessTypeAndStatus(String businessType, CodeRule.RuleStatus status) {
        LambdaQueryWrapper<CodeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CodeRuleDO::getBusinessType, businessType)
               .eq(CodeRuleDO::getStatus, status.name());
        CodeRuleDO codeRuleDO = codeRuleMapper.selectOne(wrapper);
        return Optional.ofNullable(codeRuleDO).map(this::toEntity);
    }
    
    @Override
    public List<CodeRule> findAllActive() {
        LambdaQueryWrapper<CodeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CodeRuleDO::getStatus, CodeRule.RuleStatus.ACTIVE.name());
        List<CodeRuleDO> doList = codeRuleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public CodeRule save(CodeRule codeRule) {
        CodeRuleDO codeRuleDO = toDO(codeRule);
        if (codeRule.getId() == null) {
            codeRuleMapper.insert(codeRuleDO);
            codeRule.setId(codeRuleDO.getId());
        } else {
            codeRuleMapper.updateById(codeRuleDO);
        }
        return codeRule;
    }
    
    @Override
    public void delete(Long id) {
        codeRuleMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByRuleCode(String ruleCode) {
        LambdaQueryWrapper<CodeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CodeRuleDO::getRuleCode, ruleCode);
        return codeRuleMapper.selectCount(wrapper) > 0;
    }
    
    // ========== DO 与 Entity 转换方法 ==========
    
    private CodeRule toEntity(CodeRuleDO doObj) {
        if (doObj == null) {
            return null;
        }
        CodeRule entity = new CodeRule();
        entity.setId(doObj.getId());
        entity.setRuleCode(doObj.getRuleCode());
        entity.setRuleName(doObj.getRuleName());
        entity.setBusinessType(doObj.getBusinessType());
        entity.setRuleExpression(doObj.getRuleExpression());
        entity.setStartValue(doObj.getStartValue());
        entity.setCurrentValue(doObj.getCurrentValue());
        entity.setStep(doObj.getStep());
        entity.setPaddingLength(doObj.getPaddingLength());
        entity.setStatus(CodeRule.RuleStatus.valueOf(doObj.getStatus()));
        // createdAt 和 updatedAt 通过数据库自动管理，CodeRule 实体无需手动设置
        
        // 反序列化 elements
        if (doObj.getElements() != null && !doObj.getElements().isEmpty()) {
            try {
                entity.setElements(objectMapper.readValue(doObj.getElements(), 
                    new TypeReference<List<CodeRule.RuleElement>>() {}));
            } catch (JsonProcessingException e) {
                entity.setElements(new ArrayList<>());
            }
        } else {
            entity.setElements(new ArrayList<>());
        }
        
        return entity;
    }
    
    private CodeRuleDO toDO(CodeRule entity) {
        if (entity == null) {
            return null;
        }
        CodeRuleDO doObj = new CodeRuleDO();
        doObj.setId(entity.getId());
        doObj.setRuleCode(entity.getRuleCode());
        doObj.setRuleName(entity.getRuleName());
        doObj.setBusinessType(entity.getBusinessType());
        doObj.setRuleExpression(entity.getRuleExpression());
        doObj.setStartValue(entity.getStartValue());
        doObj.setCurrentValue(entity.getCurrentValue());
        doObj.setStep(entity.getStep());
        doObj.setPaddingLength(entity.getPaddingLength());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        
        // 序列化 elements
        if (entity.getElements() != null) {
            try {
                doObj.setElements(objectMapper.writeValueAsString(entity.getElements()));
            } catch (JsonProcessingException e) {
                doObj.setElements("[]");
            }
        } else {
            doObj.setElements("[]");
        }
        
        return doObj;
    }
}