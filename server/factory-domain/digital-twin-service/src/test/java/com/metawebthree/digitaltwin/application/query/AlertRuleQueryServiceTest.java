package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertRuleLevel;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertType;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.ComparisonOperator;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.MetricType;
import com.metawebthree.digitaltwin.domain.service.AlertRuleDomainService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AlertRuleQueryServiceTest {

    @Mock
    private AlertRuleDomainService domainService;

    private AlertRuleQueryService service;

    @BeforeEach
    void setUp() {
        service = new AlertRuleQueryService(domainService);
    }

    private AlertRule createTestRule(Long id, String ruleCode, String ruleName, 
            String deviceType, MetricType metricType, boolean enabled) {
        AlertRule rule = new AlertRule();
        rule.setId(id);
        rule.setRuleCode(ruleCode);
        rule.setRuleName(ruleName);
        rule.setDeviceType(deviceType);
        rule.setMetricType(metricType);
        rule.setOperator(ComparisonOperator.GREATER_THAN);
        rule.setThresholdValue(80.0);
        rule.setLevel(AlertRuleLevel.WARNING);
        rule.setAlertType(AlertType.TEMPERATURE_HIGH);
        rule.setEnabled(enabled);
        rule.setCreatedBy("admin");
        rule.setCreatedAt(java.time.LocalDateTime.now());
        return rule;
    }

    private void assertAlertRuleDetailEquals(AlertRuleQueryService.AlertRuleDetail actual,
            Long id, String ruleCode, String ruleName, String description,
            String deviceType, String metricType, Double thresholdValue,
            Integer durationSeconds, Integer cooldownSeconds, Integer maxAlertsPerHour,
            String notificationChannels) {
        assertEquals(id, actual.id());
        assertEquals(ruleCode, actual.ruleCode());
        assertEquals(ruleName, actual.ruleName());
        assertEquals(description, actual.description());
        assertEquals(deviceType, actual.deviceType());
        assertEquals(metricType, actual.metricType());
        assertEquals(thresholdValue, actual.thresholdValue());
        assertEquals(durationSeconds, actual.durationSeconds());
        assertEquals(cooldownSeconds, actual.cooldownSeconds());
        assertEquals(maxAlertsPerHour, actual.maxAlertsPerHour());
        assertEquals(notificationChannels, actual.notificationChannels());
    }

    @Test
    void getAllRules_shouldReturnAllRules() {
        List<AlertRule> rules = Arrays.asList(
            createTestRule(1L, "RULE-001", "Temperature Alert", "SENSOR", MetricType.TEMPERATURE, true),
            createTestRule(2L, "RULE-002", "Humidity Alert", "SENSOR", MetricType.HUMIDITY, true),
            createTestRule(3L, "RULE-003", "Pressure Alert", "ACTUATOR", MetricType.PRESSURE, false)
        );
        when(domainService.getAllRules()).thenReturn(rules);

        List<AlertRuleQueryService.AlertRuleListItem> result = service.getAllRules();

        assertNotNull(result);
        assertEquals(3, result.size());
        assertEquals("RULE-001", result.get(0).ruleCode());
        assertEquals("Temperature Alert", result.get(0).ruleName());
        assertEquals("SENSOR", result.get(0).deviceType());
        verify(domainService).getAllRules();
    }

    @Test
    void getAllRules_shouldReturnEmptyListWhenNoRules() {
        when(domainService.getAllRules()).thenReturn(Collections.emptyList());

        List<AlertRuleQueryService.AlertRuleListItem> result = service.getAllRules();

        assertNotNull(result);
        assertTrue(result.isEmpty());
        verify(domainService).getAllRules();
    }

    @Test
    void getEnabledRules_shouldReturnOnlyEnabledRules() {
        List<AlertRule> enabledRules = Arrays.asList(
            createTestRule(1L, "RULE-001", "Temperature Alert", "SENSOR", MetricType.TEMPERATURE, true),
            createTestRule(2L, "RULE-002", "Humidity Alert", "SENSOR", MetricType.HUMIDITY, true)
        );
        when(domainService.getEnabledRules()).thenReturn(enabledRules);

        List<AlertRuleQueryService.AlertRuleListItem> result = service.getEnabledRules();

        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.stream().allMatch(AlertRuleQueryService.AlertRuleListItem::enabled));
        verify(domainService).getEnabledRules();
    }

    @Test
    void getRulesByDeviceType_shouldReturnFilteredRules() {
        List<AlertRule> sensorRules = Arrays.asList(
            createTestRule(1L, "RULE-001", "Temperature Alert", "SENSOR", MetricType.TEMPERATURE, true),
            createTestRule(2L, "RULE-002", "Humidity Alert", "SENSOR", MetricType.HUMIDITY, true)
        );
        when(domainService.getRulesByDeviceType("SENSOR")).thenReturn(sensorRules);

        List<AlertRuleQueryService.AlertRuleListItem> result = service.getRulesByDeviceType("SENSOR");

        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.stream().allMatch(r -> "SENSOR".equals(r.deviceType())));
        verify(domainService).getRulesByDeviceType("SENSOR");
    }

    @Test
    void getRuleById_shouldReturnRuleDetail() {
        AlertRule rule = createTestRule(1L, "RULE-001", "Temperature Alert", "SENSOR", MetricType.TEMPERATURE, true);
        rule.setDescription("Alert when temperature exceeds limit");
        rule.setTitleTemplate("Temperature exceeded: {value}°C");
        rule.setDescriptionTemplate("Current temperature {value}°C exceeds threshold");
        rule.setDurationSeconds(60);
        rule.setCooldownSeconds(300);
        rule.setMaxAlertsPerHour(10);
        rule.setNotificationChannels("EMAIL,SMS");

        when(domainService.getRuleById(1L)).thenReturn(rule);

        AlertRuleQueryService.AlertRuleDetail result = service.getRuleById(1L);

        assertNotNull(result);
        assertAlertRuleDetailEquals(result, 1L, "RULE-001", "Temperature Alert",
            "Alert when temperature exceeds limit", "SENSOR", "TEMPERATURE", 80.0,
            60, 300, 10, "EMAIL,SMS");
        verify(domainService).getRuleById(1L);
    }

    @Test
    void getRuleById_shouldReturnNullWhenNotFound() {
        when(domainService.getRuleById(999L)).thenReturn(null);

        AlertRuleQueryService.AlertRuleDetail result = service.getRuleById(999L);

        assertNull(result);
        verify(domainService).getRuleById(999L);
    }

    @Test
    void checkRuleNameUnique_shouldReturnTrueWhenNameUnique() {
        when(domainService.isRuleNameUnique("New Rule", null)).thenReturn(true);

        boolean result = service.checkRuleNameUnique("New Rule", null);

        assertTrue(result);
        verify(domainService).isRuleNameUnique("New Rule", null);
    }

    @Test
    void checkRuleNameUnique_shouldReturnFalseWhenNameExists() {
        when(domainService.isRuleNameUnique("Existing Rule", 1L)).thenReturn(false);

        boolean result = service.checkRuleNameUnique("Existing Rule", 1L);

        assertFalse(result);
        verify(domainService).isRuleNameUnique("Existing Rule", 1L);
    }

    @Test
    void checkRuleCodeUnique_shouldReturnTrueWhenCodeUnique() {
        when(domainService.isRuleCodeUnique("RULE-NEW", null)).thenReturn(true);

        boolean result = service.checkRuleCodeUnique("RULE-NEW", null);

        assertTrue(result);
        verify(domainService).isRuleCodeUnique("RULE-NEW", null);
    }

    @Test
    void checkRuleCodeUnique_shouldReturnFalseWhenCodeExists() {
        when(domainService.isRuleCodeUnique("RULE-EXISTS", 1L)).thenReturn(false);

        boolean result = service.checkRuleCodeUnique("RULE-EXISTS", 1L);

        assertFalse(result);
        verify(domainService).isRuleCodeUnique("RULE-EXISTS", 1L);
    }
}