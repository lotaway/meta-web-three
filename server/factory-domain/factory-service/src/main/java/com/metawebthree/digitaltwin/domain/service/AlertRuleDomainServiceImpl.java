package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertRuleLevel;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertType;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.ComparisonOperator;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.MetricType;
import com.metawebthree.digitaltwin.domain.repository.AlertRuleRepository;
import java.util.List;
import java.util.Objects;
import org.springframework.stereotype.Service;

@Service
public class AlertRuleDomainServiceImpl implements AlertRuleDomainService {
    private final AlertRuleRepository repository;

    public AlertRuleDomainServiceImpl(AlertRuleRepository repository) {
        this.repository = repository;
    }

    @Override
    public AlertRule createRule(String ruleCode, String ruleName, String description,
                               String deviceType, MetricType metricType, ComparisonOperator operator,
                               Double thresholdValue, AlertRuleLevel level, AlertType alertType,
                               String titleTemplate, String descriptionTemplate, String createdBy) {
        if (ruleCode == null || ruleCode.isBlank()) {
            throw new IllegalArgumentException("Rule code cannot be empty");
        }
        if (ruleName == null || ruleName.isBlank()) {
            throw new IllegalArgumentException("Rule name cannot be empty");
        }
        if (metricType == null || operator == null || thresholdValue == null) {
            throw new IllegalArgumentException("Metric type, operator, and threshold are required");
        }
        if (!isRuleCodeUnique(ruleCode, null)) {
            throw new IllegalArgumentException("Rule code already exists: " + ruleCode);
        }
        AlertRule rule = new AlertRule();
        rule.createRule(ruleCode, ruleName, description, deviceType, metricType,
                       operator, thresholdValue, level, alertType, titleTemplate,
                       descriptionTemplate, createdBy);
        return repository.save(rule);
    }

    @Override
    public AlertRule updateRule(Long id, String ruleName, String description, String deviceType,
                               MetricType metricType, ComparisonOperator operator, Double thresholdValue,
                               Integer durationSeconds, AlertRuleLevel level, AlertType alertType,
                               String titleTemplate, String descriptionTemplate,
                               Integer cooldownSeconds, Integer maxAlertsPerHour,
                               String notificationChannels, String updatedBy) {
        AlertRule rule = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Rule not found: " + id));
        rule.update(ruleName, description, deviceType, metricType, operator, thresholdValue,
                   durationSeconds, level, alertType, titleTemplate, descriptionTemplate,
                   cooldownSeconds, maxAlertsPerHour, notificationChannels, updatedBy);
        return repository.save(rule);
    }

    @Override
    public void enableRule(Long id) {
        AlertRule rule = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Rule not found: " + id));
        rule.enable();
        repository.save(rule);
    }

    @Override
    public void disableRule(Long id) {
        AlertRule rule = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Rule not found: " + id));
        rule.disable();
        repository.save(rule);
    }

    @Override
    public void deleteRule(Long id) {
        if (!repository.findById(id).isPresent()) {
            throw new IllegalArgumentException("Rule not found: " + id);
        }
        repository.deleteById(id);
    }

    @Override
    public List<AlertRule> evaluateMetrics(String deviceCode, String deviceType, String workshopId,
                                          MetricType metricType, Double currentValue, Double secondaryValue) {
        List<AlertRule> matchedRules = new java.util.ArrayList<>();
        List<AlertRule> candidateRules = repository.findByDeviceTypeAndEnabled(deviceType, true);
        for (AlertRule rule : candidateRules) {
            if ((rule.getDeviceCode() == null || rule.getDeviceCode().equals(deviceCode)) &&
                (rule.getWorkshopId() == null || rule.getWorkshopId().equals(workshopId))) {
                if (rule.evaluate(currentValue, secondaryValue)) {
                    matchedRules.add(rule);
                }
            }
        }
        return matchedRules;
    }

    @Override
    public List<AlertRule> getAllRules() {
        return repository.findAll();
    }

    @Override
    public List<AlertRule> getEnabledRules() {
        return repository.findByEnabled(true);
    }

    @Override
    public List<AlertRule> getRulesByDeviceType(String deviceType) {
        return repository.findByDeviceType(deviceType);
    }

    @Override
    public AlertRule getRuleById(Long id) {
        return repository.findById(id).orElse(null);
    }

    @Override
    public boolean isRuleNameUnique(String ruleName, Long excludeId) {
        return repository.findAll().stream()
            .filter(r -> r.getRuleName().equals(ruleName))
            .noneMatch(r -> excludeId == null || !Objects.equals(r.getId(), excludeId));
    }

    @Override
    public boolean isRuleCodeUnique(String ruleCode, Long excludeId) {
        return repository.findAll().stream()
            .filter(r -> r.getRuleCode().equals(ruleCode))
            .noneMatch(r -> excludeId == null || !Objects.equals(r.getId(), excludeId));
    }
}