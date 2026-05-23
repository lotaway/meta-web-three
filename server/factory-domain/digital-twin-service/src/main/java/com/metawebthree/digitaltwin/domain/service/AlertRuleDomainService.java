package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import java.util.List;

public interface AlertRuleDomainService {
    AlertRule createRule(String ruleCode, String ruleName, String description,
                        String deviceType, AlertRule.MetricType metricType,
                        AlertRule.ComparisonOperator operator, Double thresholdValue,
                        AlertRule.AlertRuleLevel level, AlertRule.AlertType alertType,
                        String titleTemplate, String descriptionTemplate, String createdBy);
    AlertRule updateRule(Long id, String ruleName, String description, String deviceType,
                        AlertRule.MetricType metricType, AlertRule.ComparisonOperator operator,
                        Double thresholdValue, Integer durationSeconds,
                        AlertRule.AlertRuleLevel level, AlertRule.AlertType alertType,
                        String titleTemplate, String descriptionTemplate,
                        Integer cooldownSeconds, Integer maxAlertsPerHour,
                        String notificationChannels, String updatedBy);
    void enableRule(Long id);
    void disableRule(Long id);
    void deleteRule(Long id);
    List<AlertRule> evaluateMetrics(String deviceCode, String deviceType,
                                    String workshopId, AlertRule.MetricType metricType,
                                    Double currentValue, Double secondaryValue);
    List<AlertRule> getAllRules();
    List<AlertRule> getEnabledRules();
    List<AlertRule> getRulesByDeviceType(String deviceType);
    AlertRule getRuleById(Long id);
    boolean isRuleNameUnique(String ruleName, Long excludeId);
    boolean isRuleCodeUnique(String ruleCode, Long excludeId);
}