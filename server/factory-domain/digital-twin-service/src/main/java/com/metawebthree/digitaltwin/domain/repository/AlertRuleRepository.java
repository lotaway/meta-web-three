package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import java.util.List;
import java.util.Optional;

public interface AlertRuleRepository {
    Optional<AlertRule> findById(Long id);
    Optional<AlertRule> findByRuleCode(String ruleCode);
    List<AlertRule> findAll();
    List<AlertRule> findByEnabled(Boolean enabled);
    List<AlertRule> findByDeviceType(String deviceType);
    List<AlertRule> findByWorkshopId(String workshopId);
    List<AlertRule> findByMetricType(AlertRule.MetricType metricType);
    List<AlertRule> findByDeviceTypeAndEnabled(String deviceType, Boolean enabled);
    AlertRule save(AlertRule rule);
    void deleteById(Long id);
    boolean existsByRuleCode(String ruleCode);
    Long countByEnabled(Boolean enabled);
}