package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.repository.AlertRuleRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.AlertRuleConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.AlertRuleDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.AlertRuleMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AlertRuleRepositoryImpl implements AlertRuleRepository {

    private final AlertRuleMapper alertRuleMapper;

    public AlertRuleRepositoryImpl(AlertRuleMapper alertRuleMapper) {
        this.alertRuleMapper = alertRuleMapper;
    }

    @Override
    public Optional<AlertRule> findById(Long id) {
        return Optional.ofNullable(alertRuleMapper.selectById(id))
                .map(AlertRuleConverter::toEntity);
    }

    @Override
    public Optional<AlertRule> findByRuleCode(String ruleCode) {
        AlertRuleDO d = alertRuleMapper.selectOne(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getRuleCode, ruleCode));
        return Optional.ofNullable(AlertRuleConverter.toEntity(d));
    }

    @Override
    public List<AlertRule> findAll() {
        return alertRuleMapper.selectList(null)
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<AlertRule> findByEnabled(Boolean enabled) {
        return alertRuleMapper.selectList(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getEnabled, enabled))
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<AlertRule> findByDeviceType(String deviceType) {
        return alertRuleMapper.selectList(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getDeviceType, deviceType))
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<AlertRule> findByWorkshopId(String workshopId) {
        return alertRuleMapper.selectList(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getWorkshopId, workshopId))
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<AlertRule> findByMetricType(AlertRule.MetricType metricType) {
        return alertRuleMapper.selectList(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getMetricType, metricType.name()))
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<AlertRule> findByDeviceTypeAndEnabled(String deviceType, Boolean enabled) {
        return alertRuleMapper.selectList(
                new LambdaQueryWrapper<AlertRuleDO>()
                        .eq(AlertRuleDO::getDeviceType, deviceType)
                        .eq(AlertRuleDO::getEnabled, enabled))
                .stream().map(AlertRuleConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public AlertRule save(AlertRule rule) {
        AlertRuleDO d = AlertRuleConverter.toDO(rule);
        alertRuleMapper.insert(d);
        rule.setId(d.getId());
        return rule;
    }

    @Override
    public void deleteById(Long id) {
        alertRuleMapper.deleteById(id);
    }

    @Override
    public boolean existsByRuleCode(String ruleCode) {
        return alertRuleMapper.selectCount(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getRuleCode, ruleCode)) > 0;
    }

    @Override
    public Long countByEnabled(Boolean enabled) {
        return alertRuleMapper.selectCount(
                new LambdaQueryWrapper<AlertRuleDO>().eq(AlertRuleDO::getEnabled, enabled));
    }
}
