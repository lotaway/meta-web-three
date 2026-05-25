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

    @Test
    void createRule_shouldCreateRuleSuccessfully() {
        AlertRule rule = new AlertRule();
        rule.setId(1L);
        rule.setRuleCode("RULE-001");
        rule.setRuleName("Temperature Alert");
        rule.setDeviceType("SENSOR");
        rule.setMetricType(MetricType.TEMPERATURE);
        rule.setOperator(ComparisonOperator.GREATER_THAN);
        rule.setThresholdValue(80.0);
        rule.setLevel(AlertRuleLevel.WARNING);
        rule.setAlertType(AlertType.THRESHOLD);
        rule.setEnabled(true);
        rule.setCreatedBy("admin");
        rule.setCreatedAt(java.time.LocalDateTime.now());
        rule.setUpdatedBy("admin");
        rule.setUpdatedAt(java.time.LocalDateTime.now());

        when(domainService.createRule(anyString(), anyString(), anyString(), anyString(), 
                any(), any(), anyDouble(), any(), any(), anyString(), anyString(), anyString()))
            .thenReturn(rule);

        AlertRuleCommandService.CreateAlertRuleRequest request = new AlertRuleCommandService.CreateAlertRuleRequest(
            "RULE-001", "Temperature Alert", "Alert when temperature exceeds limit",
            "SENSOR", "TEMPERATURE", "GREATER_THAN", 80.0, "WARNING", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C"
        );

        AlertRuleCommandService.AlertRuleResponse response = service.createRule(request, "admin");

        assertNotNull(response);
        assertEquals(1L, response.id());
        assertEquals("RULE-001", response.ruleCode());
        assertEquals("Temperature Alert", response.ruleName());
        verify(domainService).createRule(eq("RULE-001"), eq("Temperature Alert"), anyString(),
            eq("SENSOR"), eq(MetricType.TEMPERATURE), eq(ComparisonOperator.GREATER_THAN),
            eq(80.0), eq(AlertRuleLevel.WARNING), eq(AlertType.THRESHOLD), anyString(), anyString(), eq("admin"));
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

    @Test
    void updateRule_shouldUpdateRuleSuccessfully() {
        AlertRule rule = new AlertRule();
        rule.setId(1L);
        rule.setRuleCode("RULE-001");
        rule.setRuleName("Updated Temperature Alert");
        rule.setDeviceType("SENSOR");
        rule.setMetricType(MetricType.TEMPERATURE);
        rule.setOperator(ComparisonOperator.GREATER_THAN);
        rule.setThresholdValue(90.0);
        rule.setLevel(AlertRuleLevel.ERROR);
        rule.setAlertType(AlertType.THRESHOLD);
        rule.setEnabled(true);
        rule.setUpdatedBy("admin");
        rule.setUpdatedAt(java.time.LocalDateTime.now());

        when(domainService.updateRule(anyLong(), anyString(), anyString(), anyString(), 
                any(), any(), anyDouble(), anyInt(), any(), any(), anyString(), anyString(), 
                anyInt(), anyInt(), anyString(), anyString()))
            .thenReturn(rule);

        AlertRuleCommandService.UpdateAlertRuleRequest request = new AlertRuleCommandService.UpdateAlertRuleRequest(
            "Updated Temperature Alert", "Updated description", "SENSOR",
            "TEMPERATURE", "GREATER_THAN", 90.0, 60, "ERROR", "THRESHOLD",
            "Temperature exceeded", "Temperature is {value}°C", 300, 20, "EMAIL"
        );

        AlertRuleCommandService.AlertRuleResponse response = service.updateRule(1L, request, "admin");

        assertNotNull(response);
        assertEquals("Updated Temperature Alert", response.ruleName());
        verify(domainService).updateRule(eq(1L), anyString(), anyString(), anyString(),
            eq(MetricType.TEMPERATURE), eq(ComparisonOperator.GREATER_THAN), eq(90.0),
            eq(60), eq(AlertRuleLevel.ERROR), eq(AlertType.THRESHOLD), anyString(), anyString(),
            eq(300), eq(20), eq("EMAIL"), eq("admin"));
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