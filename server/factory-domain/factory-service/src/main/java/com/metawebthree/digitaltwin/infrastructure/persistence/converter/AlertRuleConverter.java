package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.AlertRuleDO;
import java.sql.Timestamp;

public class AlertRuleConverter {

    public static AlertRuleDO toDO(AlertRule entity) {
        if (entity == null) return null;
        AlertRuleDO d = new AlertRuleDO();
        d.setId(entity.getId());
        d.setRuleCode(entity.getRuleCode());
        d.setRuleName(entity.getRuleName());
        d.setDescription(entity.getDescription());
        d.setDeviceType(entity.getDeviceType());
        d.setDeviceCode(entity.getDeviceCode());
        d.setWorkshopId(entity.getWorkshopId());
        d.setMetricType(entity.getMetricType() != null ? entity.getMetricType().name() : null);
        d.setOperator(entity.getOperator() != null ? entity.getOperator().name() : null);
        d.setThresholdValue(entity.getThresholdValue());
        d.setDurationSeconds(entity.getDurationSeconds());
        d.setLevel(entity.getLevel() != null ? entity.getLevel().name() : null);
        d.setAlertType(entity.getAlertType() != null ? entity.getAlertType().name() : null);
        d.setTitleTemplate(entity.getTitleTemplate());
        d.setDescriptionTemplate(entity.getDescriptionTemplate());
        d.setEnabled(entity.getEnabled());
        d.setCooldownSeconds(entity.getCooldownSeconds());
        d.setMaxAlertsPerHour(entity.getMaxAlertsPerHour());
        d.setNotificationChannels(entity.getNotificationChannels());
        d.setCreatedBy(entity.getCreatedBy());
        d.setUpdatedBy(entity.getUpdatedBy());
        if (entity.getCreatedAt() != null) d.setCreatedAt(Timestamp.valueOf(entity.getCreatedAt()));
        if (entity.getUpdatedAt() != null) d.setUpdatedAt(Timestamp.valueOf(entity.getUpdatedAt()));
        return d;
    }

    public static AlertRule toEntity(AlertRuleDO d) {
        if (d == null) return null;
        AlertRule entity = new AlertRule();
        entity.setId(d.getId());
        entity.setRuleCode(d.getRuleCode());
        entity.setRuleName(d.getRuleName());
        entity.setDescription(d.getDescription());
        entity.setDeviceType(d.getDeviceType());
        entity.setDeviceCode(d.getDeviceCode());
        entity.setWorkshopId(d.getWorkshopId());
        if (d.getMetricType() != null) entity.setMetricType(AlertRule.MetricType.valueOf(d.getMetricType()));
        if (d.getOperator() != null) entity.setOperator(AlertRule.ComparisonOperator.valueOf(d.getOperator()));
        entity.setThresholdValue(d.getThresholdValue());
        entity.setDurationSeconds(d.getDurationSeconds());
        if (d.getLevel() != null) entity.setLevel(AlertRule.AlertRuleLevel.valueOf(d.getLevel()));
        if (d.getAlertType() != null) entity.setAlertType(AlertRule.AlertType.valueOf(d.getAlertType()));
        entity.setTitleTemplate(d.getTitleTemplate());
        entity.setDescriptionTemplate(d.getDescriptionTemplate());
        entity.setEnabled(d.getEnabled());
        entity.setCooldownSeconds(d.getCooldownSeconds());
        entity.setMaxAlertsPerHour(d.getMaxAlertsPerHour());
        entity.setNotificationChannels(d.getNotificationChannels());
        entity.setCreatedBy(d.getCreatedBy());
        entity.setUpdatedBy(d.getUpdatedBy());
        if (d.getCreatedAt() != null) entity.setCreatedAt(d.getCreatedAt().toLocalDateTime());
        if (d.getUpdatedAt() != null) entity.setUpdatedAt(d.getUpdatedAt().toLocalDateTime());
        return entity;
    }
}
