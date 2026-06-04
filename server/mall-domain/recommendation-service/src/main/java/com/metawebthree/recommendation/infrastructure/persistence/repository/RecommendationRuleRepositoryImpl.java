package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationRuleMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class RecommendationRuleRepositoryImpl implements RecommendationRuleRepository {

    private final RecommendationRuleMapper recommendationRuleMapper;

    public RecommendationRuleRepositoryImpl(RecommendationRuleMapper recommendationRuleMapper) {
        this.recommendationRuleMapper = recommendationRuleMapper;
    }

    @Override
    public Optional<RecommendationRule> findById(Long id) {
        return Optional.ofNullable(recommendationRuleMapper.selectById(id))
            .map(this::toDomain);
    }

    @Override
    public List<RecommendationRule> findByScene(String scene) {
        LambdaQueryWrapper<RecommendationRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationRuleDO::getScene, scene);
        return recommendationRuleMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationRule> findByStatus(RecommendationRule.RuleStatus status) {
        LambdaQueryWrapper<RecommendationRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationRuleDO::getStatus, status.name());
        return recommendationRuleMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationRule> findBySceneAndStatus(String scene, RecommendationRule.RuleStatus status) {
        LambdaQueryWrapper<RecommendationRuleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationRuleDO::getScene, scene)
            .eq(RecommendationRuleDO::getStatus, status.name());
        return recommendationRuleMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public RecommendationRule save(RecommendationRule rule) {
        RecommendationRuleDO ruleDO = toDO(rule);
        if (rule.getId() == null) {
            recommendationRuleMapper.insert(ruleDO);
            rule.setId(ruleDO.getId());
        } else {
            recommendationRuleMapper.updateById(ruleDO);
        }
        return rule;
    }

    @Override
    public void update(RecommendationRule rule) {
        recommendationRuleMapper.updateById(toDO(rule));
    }

    @Override
    public List<RecommendationRule> findAll() {
        return recommendationRuleMapper.selectList(null).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationRule> findActiveRules() {
        return findByStatus(RecommendationRule.RuleStatus.ACTIVE);
    }

    @Override
    public long count() {
        return recommendationRuleMapper.selectCount(null);
    }

    @Override
    public void deleteById(Long id) {
        recommendationRuleMapper.deleteById(id);
    }

    private RecommendationRule toDomain(RecommendationRuleDO ruleDO) {
        RecommendationRule rule = new RecommendationRule();
        rule.setId(ruleDO.getId());
        rule.setRuleName(ruleDO.getRuleName());
        rule.setScene(ruleDO.getScene());
        rule.setType(RecommendationRule.RuleType.valueOf(ruleDO.getType()));
        rule.setStatus(RecommendationRule.RuleStatus.valueOf(ruleDO.getStatus()));
        rule.setPriority(ruleDO.getPriority());
        rule.setMaxItems(ruleDO.getMaxItems());
        rule.setMinScore(ruleDO.getMinScore());
        rule.setConditions(ruleDO.getConditions());
        rule.setExclusions(ruleDO.getExclusions());
        rule.setBoostFactor(ruleDO.getBoostFactor());

        if (ruleDO.getTargetSkus() != null && !ruleDO.getTargetSkus().isEmpty()) {
            rule.setTargetSkus(Arrays.asList(ruleDO.getTargetSkus().split(",")));
        } else {
            rule.setTargetSkus(Collections.emptyList());
        }
        return rule;
    }

    private RecommendationRuleDO toDO(RecommendationRule rule) {
        RecommendationRuleDO ruleDO = new RecommendationRuleDO();
        ruleDO.setId(rule.getId());
        ruleDO.setRuleName(rule.getRuleName());
        ruleDO.setScene(rule.getScene());
        ruleDO.setType(rule.getType().name());
        ruleDO.setStatus(rule.getStatus().name());
        ruleDO.setPriority(rule.getPriority());
        ruleDO.setMaxItems(rule.getMaxItems());
        ruleDO.setMinScore(rule.getMinScore());
        ruleDO.setConditions(rule.getConditions());
        ruleDO.setExclusions(rule.getExclusions());
        ruleDO.setBoostFactor(rule.getBoostFactor());
        ruleDO.setCreatedAt(rule.getCreatedAt());
        ruleDO.setUpdatedAt(rule.getUpdatedAt());
        if (rule.getTargetSkus() != null) {
            ruleDO.setTargetSkus(String.join(",", rule.getTargetSkus()));
        }
        return ruleDO;
    }
}
