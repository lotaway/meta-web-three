package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertRuleLevel;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertType;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.ComparisonOperator;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.MetricType;
import com.metawebthree.digitaltwin.domain.service.AlertRuleDomainService;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
public class AlertRuleCommandService {
    private final AlertRuleDomainService domainService;

    public AlertRuleCommandService(AlertRuleDomainService domainService) {
        this.domainService = domainService;
    }

    public record CreateAlertRuleRequest(
        String ruleCode,
        String ruleName,
        String description,
        String deviceType,
        String metricType,
        String operator,
        Double thresholdValue,
        String level,
        String alertType,
        String titleTemplate,
        String descriptionTemplate
    ) {}

    public record UpdateAlertRuleRequest(
        String ruleName,
        String description,
        String deviceType,
        String metricType,
        String operator,
        Double thresholdValue,
        Integer durationSeconds,
        String level,
        String alertType,
        String titleTemplate,
        String descriptionTemplate,
        Integer cooldownSeconds,
        Integer maxAlertsPerHour,
        String notificationChannels
    ) {}

    public record AlertRuleResponse(
        Long id,
        String ruleCode,
        String ruleName,
        String description,
        String deviceType,
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

    public AlertRuleResponse createRule(CreateAlertRuleRequest request, String userId) {
        validateCreateRequest(request);
        AlertRule rule = domainService.createRule(
            request.ruleCode(),
            request.ruleName(),
            request.description(),
            request.deviceType(),
            MetricType.valueOf(request.metricType()),
            ComparisonOperator.valueOf(request.operator()),
            request.thresholdValue(),
            AlertRuleLevel.valueOf(request.level()),
            AlertType.valueOf(request.alertType()),
            request.titleTemplate(),
            request.descriptionTemplate(),
            userId
        );
        return toResponse(rule);
    }

    public AlertRuleResponse updateRule(Long id, UpdateAlertRuleRequest request, String userId) {
        validateUpdateRequest(request);
        AlertRule rule = domainService.updateRule(
            id,
            request.ruleName(),
            request.description(),
            request.deviceType(),
            MetricType.valueOf(request.metricType()),
            ComparisonOperator.valueOf(request.operator()),
            request.thresholdValue(),
            request.durationSeconds(),
            AlertRuleLevel.valueOf(request.level()),
            AlertType.valueOf(request.alertType()),
            request.titleTemplate(),
            request.descriptionTemplate(),
            request.cooldownSeconds(),
            request.maxAlertsPerHour(),
            request.notificationChannels(),
            userId
        );
        return toResponse(rule);
    }

    public Map<String, Object> enableRule(Long id) {
        domainService.enableRule(id);
        return Map.of("success", true, "message", "Rule enabled successfully");
    }

    public Map<String, Object> disableRule(Long id) {
        domainService.disableRule(id);
        return Map.of("success", true, "message", "Rule disabled successfully");
    }

    public Map<String, Object> deleteRule(Long id) {
        domainService.deleteRule(id);
        return Map.of("success", true, "message", "Rule deleted successfully");
    }

    private void validateCreateRequest(CreateAlertRuleRequest request) {
        if (request.ruleCode() == null || request.ruleCode().isBlank()) {
            throw new IllegalArgumentException("Rule code is required");
        }
        if (request.ruleName() == null || request.ruleName().isBlank()) {
            throw new IllegalArgumentException("Rule name is required");
        }
        if (request.thresholdValue() == null) {
            throw new IllegalArgumentException("Threshold value is required");
        }
    }

    private void validateUpdateRequest(UpdateAlertRuleRequest request) {
        if (request.ruleName() == null || request.ruleName().isBlank()) {
            throw new IllegalArgumentException("Rule name is required");
        }
    }

    private AlertRuleResponse toResponse(AlertRule rule) {
        return new AlertRuleResponse(
            rule.getId(),
            rule.getRuleCode(),
            rule.getRuleName(),
            rule.getDescription(),
            rule.getDeviceType(),
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