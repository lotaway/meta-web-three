package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.service.AlertRuleDomainService;
import java.util.List;

public class AlertRuleQueryService {
    private final AlertRuleDomainService domainService;

    public AlertRuleQueryService(AlertRuleDomainService domainService) {
        this.domainService = domainService;
    }

    public record AlertRuleListItem(
        Long id,
        String ruleCode,
        String ruleName,
        String deviceType,
        String metricType,
        String operator,
        Double thresholdValue,
        String level,
        Boolean enabled,
        String createdAt
    ) {}

    public record AlertRuleDetail(
        Long id,
        String ruleCode,
        String ruleName,
        String description,
        String deviceType,
        String deviceCode,
        String workshopId,
        String metricType,
        String operator,
        Double thresholdValue,
        Integer durationSeconds,
        String level,
        String alertType,
        String titleTemplate,
        String descriptionTemplate,
        Boolean enabled,
        Integer cooldownSeconds,
        Integer maxAlertsPerHour,
        String notificationChannels,
        String createdBy,
        String createdAt,
        String updatedBy,
        String updatedAt
    ) {}

    public List<AlertRuleListItem> getAllRules() {
        return domainService.getAllRules().stream()
            .map(this::toListItem)
            .toList();
    }

    public List<AlertRuleListItem> getEnabledRules() {
        return domainService.getEnabledRules().stream()
            .map(this::toListItem)
            .toList();
    }

    public List<AlertRuleListItem> getRulesByDeviceType(String deviceType) {
        return domainService.getRulesByDeviceType(deviceType).stream()
            .map(this::toListItem)
            .toList();
    }

    public AlertRuleDetail getRuleById(Long id) {
        AlertRule rule = domainService.getRuleById(id);
        if (rule == null) {
            return null;
        }
        return toDetail(rule);
    }

    public boolean checkRuleNameUnique(String ruleName, Long excludeId) {
        return domainService.isRuleNameUnique(ruleName, excludeId);
    }

    public boolean checkRuleCodeUnique(String ruleCode, Long excludeId) {
        return domainService.isRuleCodeUnique(ruleCode, excludeId);
    }

    private AlertRuleListItem toListItem(AlertRule rule) {
        return new AlertRuleListItem(
            rule.getId(),
            rule.getRuleCode(),
            rule.getRuleName(),
            rule.getDeviceType(),
            rule.getMetricType() != null ? rule.getMetricType().name() : null,
            rule.getOperator() != null ? rule.getOperator().name() : null,
            rule.getThresholdValue(),
            rule.getLevel() != null ? rule.getLevel().name() : null,
            rule.getEnabled(),
            rule.getCreatedAt() != null ? rule.getCreatedAt().toString() : null
        );
    }

    private AlertRuleDetail toDetail(AlertRule rule) {
        return new AlertRuleDetail(
            rule.getId(),
            rule.getRuleCode(),
            rule.getRuleName(),
            rule.getDescription(),
            rule.getDeviceType(),
            rule.getDeviceCode(),
            rule.getWorkshopId(),
            rule.getMetricType() != null ? rule.getMetricType().name() : null,
            rule.getOperator() != null ? rule.getOperator().name() : null,
            rule.getThresholdValue(),
            rule.getDurationSeconds(),
            rule.getLevel() != null ? rule.getLevel().name() : null,
            rule.getAlertType() != null ? rule.getAlertType().name() : null,
            rule.getTitleTemplate(),
            rule.getDescriptionTemplate(),
            rule.getEnabled(),
            rule.getCooldownSeconds(),
            rule.getMaxAlertsPerHour(),
            rule.getNotificationChannels(),
            rule.getCreatedBy(),
            rule.getCreatedAt() != null ? rule.getCreatedAt().toString() : null,
            rule.getUpdatedBy(),
            rule.getUpdatedAt() != null ? rule.getUpdatedAt().toString() : null
        );
    }
}