package com.metawebthree.dom.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.dom.domain.entity.SourcingRule;
import com.metawebthree.dom.domain.repository.SourcingRuleRepository;
import com.metawebthree.dom.infrastructure.persistence.converter.SourcingRuleConverter;
import com.metawebthree.dom.infrastructure.persistence.dataobject.SourcingRuleDO;
import com.metawebthree.dom.infrastructure.persistence.mapper.SourcingRuleMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class SourcingRuleRepositoryImpl implements SourcingRuleRepository {

    private final SourcingRuleMapper sourcingRuleMapper;
    private final SourcingRuleConverter sourcingRuleConverter;

    public SourcingRuleRepositoryImpl(SourcingRuleMapper sourcingRuleMapper, SourcingRuleConverter sourcingRuleConverter) {
        this.sourcingRuleMapper = sourcingRuleMapper;
        this.sourcingRuleConverter = sourcingRuleConverter;
    }

    @Override
    public Optional<SourcingRule> findById(Long id) {
        SourcingRuleDO doObj = sourcingRuleMapper.selectById(id);
        return Optional.ofNullable(sourcingRuleConverter.toEntity(doObj));
    }

    @Override
    public List<SourcingRule> findByRuleType(String ruleType) {
        LambdaQueryWrapper<SourcingRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SourcingRuleDO::getRuleType, ruleType);
        return sourcingRuleMapper.selectList(wrapper).stream()
                .map(sourcingRuleConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<SourcingRule> findByRegion(String region) {
        LambdaQueryWrapper<SourcingRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SourcingRuleDO::getRegion, region);
        return sourcingRuleMapper.selectList(wrapper).stream()
                .map(sourcingRuleConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<SourcingRule> findByEnabled(Boolean enabled) {
        LambdaQueryWrapper<SourcingRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SourcingRuleDO::getEnabled, enabled);
        return sourcingRuleMapper.selectList(wrapper).stream()
                .map(sourcingRuleConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<SourcingRule> findAll() {
        return sourcingRuleMapper.selectList(null).stream()
                .map(sourcingRuleConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public SourcingRule save(SourcingRule rule) {
        SourcingRuleDO doObj = sourcingRuleConverter.toDO(rule);
        if (rule.getId() == null) {
            sourcingRuleMapper.insert(doObj);
            rule.setId(doObj.getId());
        } else {
            sourcingRuleMapper.updateById(doObj);
        }
        return rule;
    }

    @Override
    public void delete(SourcingRule rule) {
        if (rule.getId() != null) {
            sourcingRuleMapper.deleteById(rule.getId());
        }
    }
}
