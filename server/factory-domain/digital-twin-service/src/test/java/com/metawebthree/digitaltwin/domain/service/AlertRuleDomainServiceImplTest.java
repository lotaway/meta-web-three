package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.AlertRule;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertRuleLevel;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.AlertType;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.ComparisonOperator;
import com.metawebthree.digitaltwin.domain.entity.AlertRule.MetricType;
import com.metawebthree.digitaltwin.domain.repository.AlertRuleRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AlertRuleDomainServiceImplTest {
    @Mock
    private AlertRuleRepository repository;
    private AlertRuleDomainServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new AlertRuleDomainServiceImpl(repository);
    }

    @Test
    void createRule_shouldCreateSuccessfully() {
        AlertRule savedRule = new AlertRule();
        savedRule.setId(1L);
        savedRule.setRuleCode("TEMP-HIGH-001");
        savedRule.setRuleName("Temperature High Alert");
        savedRule.setDeviceType("SENSOR");
        savedRule.setMetricType(MetricType.TEMPERATURE);
        savedRule.setOperator(ComparisonOperator.GREATER_THAN);
        savedRule.setThresholdValue(80.0);
        savedRule.setLevel(AlertRuleLevel.WARNING);
        savedRule.setAlertType(AlertType.TEMPERATURE_HIGH);
        savedRule.setEnabled(true);

        when(repository.findAll()).thenReturn(List.of());
        when(repository.save(any(AlertRule.class))).thenReturn(savedRule);

        AlertRule result = service.createRule(
            "TEMP-HIGH-001",
            "Temperature High Alert",
            "Alert when temperature exceeds threshold",
            "SENSOR",
            MetricType.TEMPERATURE,
            ComparisonOperator.GREATER_THAN,
            80.0,
            AlertRuleLevel.WARNING,
            AlertType.TEMPERATURE_HIGH,
            "Temperature high for {device}",
            "Temperature {value} exceeds threshold {threshold}",
            "admin"
        );

        assertNotNull(result);
        assertEquals("TEMP-HIGH-001", result.getRuleCode());
        assertEquals("Temperature High Alert", result.getRuleName());
        assertTrue(result.getEnabled());
        verify(repository).save(any(AlertRule.class));
    }

    @Test
    void createRule_shouldThrowExceptionForDuplicateCode() {
        AlertRule existingRule = new AlertRule();
        existingRule.setRuleCode("TEMP-HIGH-001");
        when(repository.findAll()).thenReturn(List.of(existingRule));

        assertThrows(IllegalArgumentException.class, () ->
            service.createRule(
                "TEMP-HIGH-001",
                "New Rule",
                "Description",
                "SENSOR",
                MetricType.TEMPERATURE,
                ComparisonOperator.GREATER_THAN,
                80.0,
                AlertRuleLevel.WARNING,
                AlertType.TEMPERATURE_HIGH,
                "Title",
                "Desc",
                "admin"
            )
        );
    }

    @Test
    void createRule_shouldThrowExceptionForEmptyCode() {
        assertThrows(IllegalArgumentException.class, () ->
            service.createRule(
                "",
                "Rule Name",
                "Description",
                "SENSOR",
                MetricType.TEMPERATURE,
                ComparisonOperator.GREATER_THAN,
                80.0,
                AlertRuleLevel.WARNING,
                AlertType.TEMPERATURE_HIGH,
                "Title",
                "Desc",
                "admin"
            )
        );
    }

    @Test
    void evaluateMetrics_shouldMatchCorrectRules() {
        AlertRule rule1 = createTestRule(1L, "SENSOR", MetricType.TEMPERATURE, 
            ComparisonOperator.GREATER_THAN, 80.0);
        AlertRule rule2 = createTestRule(2L, "SENSOR", MetricType.TEMPERATURE, 
            ComparisonOperator.LESS_THAN, 10.0);
        
        when(repository.findByDeviceTypeAndEnabled("SENSOR", true))
            .thenReturn(List.of(rule1, rule2));

        List<AlertRule> matched = service.evaluateMetrics(
            "SENSOR-001", "SENSOR", "WS001", MetricType.TEMPERATURE, 85.0, null
        );

        assertEquals(1, matched.size());
        assertEquals(1L, matched.get(0).getId());
    }

    @Test
    void evaluateMetrics_shouldNotMatchDisabledRules() {
        AlertRule rule = createTestRule(1L, "SENSOR", MetricType.TEMPERATURE, 
            ComparisonOperator.GREATER_THAN, 80.0);
        rule.disable();
        
        when(repository.findByDeviceTypeAndEnabled("SENSOR", true))
            .thenReturn(List.of());

        List<AlertRule> matched = service.evaluateMetrics(
            "SENSOR-001", "SENSOR", "WS001", MetricType.TEMPERATURE, 85.0, null
        );

        assertEquals(0, matched.size());
    }

    @Test
    void enableRule_shouldEnableDisabledRule() {
        AlertRule rule = new AlertRule();
        rule.setId(1L);
        rule.setEnabled(false);
        
        when(repository.findById(1L)).thenReturn(Optional.of(rule));
        when(repository.save(any(AlertRule.class))).thenReturn(rule);

        service.enableRule(1L);

        assertTrue(rule.getEnabled());
        verify(repository).save(rule);
    }

    @Test
    void disableRule_shouldDisableEnabledRule() {
        AlertRule rule = new AlertRule();
        rule.setId(1L);
        rule.setEnabled(true);
        
        when(repository.findById(1L)).thenReturn(Optional.of(rule));
        when(repository.save(any(AlertRule.class))).thenReturn(rule);

        service.disableRule(1L);

        assertFalse(rule.getEnabled());
        verify(repository).save(rule);
    }

    @Test
    void deleteRule_shouldDeleteExistingRule() {
        AlertRule rule = new AlertRule();
        rule.setId(1L);
        
        when(repository.findById(1L)).thenReturn(Optional.of(rule));
        
        service.deleteRule(1L);

        verify(repository).deleteById(1L);
    }

    @Test
    void deleteRule_shouldThrowExceptionForNonExistentRule() {
        when(repository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () ->
            service.deleteRule(1L)
        );
    }

    @Test
    void getAllRules_shouldReturnAllRules() {
        AlertRule rule1 = new AlertRule();
        rule1.setId(1L);
        AlertRule rule2 = new AlertRule();
        rule2.setId(2L);
        
        when(repository.findAll()).thenReturn(List.of(rule1, rule2));

        List<AlertRule> result = service.getAllRules();

        assertEquals(2, result.size());
    }

    @Test
    void getEnabledRules_shouldReturnOnlyEnabledRules() {
        AlertRule rule1 = new AlertRule();
        rule1.setEnabled(true);
        AlertRule rule2 = new AlertRule();
        rule2.setEnabled(false);
        
        when(repository.findByEnabled(true)).thenReturn(List.of(rule1));

        List<AlertRule> result = service.getEnabledRules();

        assertEquals(1, result.size());
    }

    private AlertRule createTestRule(Long id, String deviceType, MetricType metricType,
                                     ComparisonOperator operator, Double threshold) {
        AlertRule rule = new AlertRule();
        rule.setId(id);
        rule.setDeviceType(deviceType);
        rule.setMetricType(metricType);
        rule.setOperator(operator);
        rule.setThresholdValue(threshold);
        rule.setEnabled(true);
        return rule;
    }
}