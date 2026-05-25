package com.metawebthree.digitaltwin.application.command;

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

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AlertRuleCommandServiceTest {

    @Mock
    private AlertRuleDomainService domainService;

    private AlertRuleCommandService service;

    @BeforeEach
    void setUp() {
        service = new AlertRuleCommandService(domainService);
    }

    private AlertRule createSampleRule(Long id, String ruleCode) {
        AlertRule rule = new AlertRule();
        rule.setId(id);
        rule.setRuleCode(ruleCode);
        rule.setRuleName(ruleCode + " Alert");
        rule.setDeviceType("SENSOR");
        rule.setMetricType(MetricType.TEMPERATURE);
        rule.setOperator(ComparisonOperator.GREATER_THAN);
        rule.setThresholdValue(80.0);
        rule.setLevel(AlertRuleLevel.WARNING);
        rule.setAlertType(AlertType.THRESHOLD);
        rule.setEnabled(true);
        rule.setCreatedBy("admin");
        return rule;
    }

    private AlertRuleCommandService.CreateAlertRuleRequest createBaseRequest() {
        return new AlertRuleCommandService.CreateAlertRuleRequest(
            "RULE-001", "Temperature Alert", "Alert when temperature exceeds limit",
            "SENSOR", "TEMPERATURE", "GREATER_THAN", 80.0, "WARNING", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C");
    }

    private void prepareCreateMock(AlertRule rule) {
        when(domainService.createRule(anyString(), anyString(), anyString(), anyString(),
            any(AlertRule.MetricType.class), any(AlertRule.ComparisonOperator.class), anyDouble(),
            any(AlertRule.AlertRuleLevel.class), any(AlertRule.AlertType.class),
            anyString(), anyString(), anyString(), anyString()))
            .thenReturn(rule);
    }

    @Test
    void createRule_shouldCreateRuleSuccessfully() {
        AlertRule rule = createSampleRule(1L, "RULE-001");
        prepareCreateMock(rule);

        AlertRuleCommandService.AlertRuleResponse response = service.createRule(createBaseRequest(), "admin");

        assertNotNull(response);
        assertEquals(1L, response.id());
        assertEquals("RULE-001", response.ruleCode());
        assertEquals("Temperature Alert", response.ruleName());
    }

    @Test
    void createRule_shouldThrowExceptionWhenRuleCodeBlank() {
        AlertRuleCommandService.CreateAlertRuleRequest request = new AlertRuleCommandService.CreateAlertRuleRequest(
            "", "Temperature Alert", "Alert when temperature exceeds limit",
            "SENSOR", "TEMPERATURE", "GREATER_THAN", 80.0, "WARNING", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C"
        );

        assertThrows(IllegalArgumentException.class, () -> service.createRule(request, "admin"));
    }

    @Test
    void createRule_shouldThrowExceptionWhenRuleNameBlank() {
        AlertRuleCommandService.CreateAlertRuleRequest request = new AlertRuleCommandService.CreateAlertRuleRequest(
            "RULE-001", "", "Alert when temperature exceeds limit",
            "SENSOR", "TEMPERATURE", "GREATER_THAN", 80.0, "WARNING", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C"
        );

        assertThrows(IllegalArgumentException.class, () -> service.createRule(request, "admin"));
    }

    @Test
    void createRule_shouldThrowExceptionWhenThresholdNull() {
        AlertRuleCommandService.CreateAlertRuleRequest request = new AlertRuleCommandService.CreateAlertRuleRequest(
            "RULE-001", "Temperature Alert", "Alert when temperature exceeds limit",
            "SENSOR", "TEMPERATURE", "GREATER_THAN", null, "WARNING", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C"
        );

        assertThrows(IllegalArgumentException.class, () -> service.createRule(request, "admin"));
    }

    private void prepareUpdateMock(AlertRule rule) {
        when(domainService.updateRule(anyLong(), anyString(), anyString(), anyString(),
            any(AlertRule.MetricType.class), any(AlertRule.ComparisonOperator.class), anyDouble(), anyInt(),
            any(AlertRule.AlertRuleLevel.class), any(AlertRule.AlertType.class), anyString(), anyString(),
            anyInt(), anyInt(), anyString(), anyString()))
            .thenReturn(rule);
    }

    private AlertRuleCommandService.UpdateAlertRuleRequest createUpdateRequest() {
        return new AlertRuleCommandService.UpdateAlertRuleRequest(
            "Updated Temperature Alert", "Updated description", "SENSOR",
            "TEMPERATURE", "GREATER_THAN", 90.0, 60, "ERROR", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C", 300, 20, "EMAIL");
    }

    @Test
    void updateRule_shouldUpdateRuleSuccessfully() {
        AlertRule rule = createSampleRule(1L, "RULE-001");
        rule.setRuleName("Updated Temperature Alert");
        rule.setThresholdValue(90.0);
        rule.setLevel(AlertRuleLevel.ERROR);
        prepareUpdateMock(rule);

        AlertRuleCommandService.AlertRuleResponse response = service.updateRule(1L, createUpdateRequest(), "admin");

        assertNotNull(response);
        assertEquals("Updated Temperature Alert", response.ruleName());
    }

    @Test
    void enableRule_shouldEnableRuleSuccessfully() {
        doNothing().when(domainService).enableRule(anyLong());

        Map<String, Object> result = service.enableRule(1L);

        assertNotNull(result);
        assertEquals(true, result.get("success"));
        verify(domainService).enableRule(1L);
    }

    @Test
    void disableRule_shouldDisableRuleSuccessfully() {
        doNothing().when(domainService).disableRule(anyLong());

        Map<String, Object> result = service.disableRule(1L);

        assertNotNull(result);
        assertEquals(true, result.get("success"));
        verify(domainService).disableRule(1L);
    }

    @Test
    void deleteRule_shouldDeleteRuleSuccessfully() {
        doNothing().when(domainService).deleteRule(anyLong());

        Map<String, Object> result = service.deleteRule(1L);

        assertNotNull(result);
        assertEquals(true, result.get("success"));
        verify(domainService).deleteRule(1L);
    }
}