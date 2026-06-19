package com.metawebthree.mes.application.command;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.PokayokeRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.PokayokeRuleMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class PokayokeRuleService {
    
    private final PokayokeRuleMapper ruleMapper;
    private final ObjectMapper objectMapper;
    
    @Transactional
    public PokayokeRuleDO createRule(PokayokeRuleDO rule) {
        rule.setStatus("DRAFT");
        rule.setEnabled(true);
        rule.setDeleted(false);
        rule.setCreatedAt(LocalDateTime.now());
        ruleMapper.insert(rule);
        return rule;
    }
    
    @Transactional
    public PokayokeRuleDO updateRule(PokayokeRuleDO rule) {
        rule.setUpdatedAt(LocalDateTime.now());
        ruleMapper.updateById(rule);
        return rule;
    }
    
    @Transactional
    public void activateRule(Long ruleId) {
        PokayokeRuleDO rule = ruleMapper.selectById(ruleId);
        if (rule != null) {
            rule.setStatus("ACTIVE");
            rule.setUpdatedAt(LocalDateTime.now());
            ruleMapper.updateById(rule);
        }
    }
    
    @Transactional
    public void deactivateRule(Long ruleId) {
        PokayokeRuleDO rule = ruleMapper.selectById(ruleId);
        if (rule != null) {
            rule.setStatus("INACTIVE");
            rule.setUpdatedAt(LocalDateTime.now());
            ruleMapper.updateById(rule);
        }
    }
    
    @Transactional
    public void deleteRule(Long ruleId) {
        PokayokeRuleDO rule = ruleMapper.selectById(ruleId);
        if (rule != null) {
            rule.setDeleted(true);
            rule.setUpdatedAt(LocalDateTime.now());
            ruleMapper.updateById(rule);
        }
    }
    
    public PokayokeRuleDO getRule(Long ruleId) {
        return ruleMapper.selectById(ruleId);
    }
    
    public List<PokayokeRuleDO> listRules(String status, String ruleType) {
        LambdaQueryWrapper<PokayokeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(PokayokeRuleDO::getDeleted, false);
        if (status != null && !status.isEmpty()) {
            wrapper.eq(PokayokeRuleDO::getStatus, status);
        }
        if (ruleType != null && !ruleType.isEmpty()) {
            wrapper.eq(PokayokeRuleDO::getRuleType, ruleType);
        }
        wrapper.orderByDesc(PokayokeRuleDO::getCreatedAt);
        return ruleMapper.selectList(wrapper);
    }
    
    public List<PokayokeRuleDO> getActiveRules() {
        LambdaQueryWrapper<PokayokeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(PokayokeRuleDO::getDeleted, false)
               .eq(PokayokeRuleDO::getStatus, "ACTIVE")
               .eq(PokayokeRuleDO::getEnabled, true);
        return ruleMapper.selectList(wrapper);
    }
    
    public List<PokayokeRuleDO> getRulesByWorkstation(String workstationId) {
        LambdaQueryWrapper<PokayokeRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(PokayokeRuleDO::getDeleted, false)
               .eq(PokayokeRuleDO::getWorkstationId, workstationId)
               .eq(PokayokeRuleDO::getEnabled, true);
        return ruleMapper.selectList(wrapper);
    }
}