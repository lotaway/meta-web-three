package com.metawebthree.digitaltwin.interfaces.controller;

import com.metawebthree.digitaltwin.application.command.AlertRuleCommandService;
import com.metawebthree.digitaltwin.application.command.AlertRuleCommandService.AlertRuleResponse;
import com.metawebthree.digitaltwin.application.command.AlertRuleCommandService.CreateAlertRuleRequest;
import com.metawebthree.digitaltwin.application.command.AlertRuleCommandService.UpdateAlertRuleRequest;
import com.metawebthree.digitaltwin.application.query.AlertRuleQueryService;
import com.metawebthree.digitaltwin.application.query.AlertRuleQueryService.AlertRuleDetail;
import com.metawebthree.digitaltwin.application.query.AlertRuleQueryService.AlertRuleListItem;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/alert-rules")
public class AlertRuleController {
    private final AlertRuleCommandService commandService;
    private final AlertRuleQueryService queryService;

    public AlertRuleController(AlertRuleCommandService commandService,
                              AlertRuleQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @GetMapping
    public ResponseEntity<List<AlertRuleListItem>> getAllRules(
            @RequestParam(required = false) Boolean enabled,
            @RequestParam(required = false) String deviceType) {
        List<AlertRuleListItem> rules;
        if (Boolean.TRUE.equals(enabled)) {
            rules = queryService.getEnabledRules();
        } else if (deviceType != null && !deviceType.isBlank()) {
            rules = queryService.getRulesByDeviceType(deviceType);
        } else {
            rules = queryService.getAllRules();
        }
        return ResponseEntity.ok(rules);
    }

    @GetMapping("/{id}")
    public ResponseEntity<AlertRuleDetail> getRuleById(@PathVariable Long id) {
        AlertRuleDetail rule = queryService.getRuleById(id);
        if (rule == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(rule);
    }

    @PostMapping
    public ResponseEntity<AlertRuleResponse> createRule(
            @RequestBody CreateAlertRuleRequest request,
            @RequestHeader(value = "X-Operator-Id", defaultValue = "system") String operatorId) {
        try {
            AlertRuleResponse rule = commandService.createRule(request, operatorId);
            return ResponseEntity.ok(rule);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PutMapping("/{id}")
    public ResponseEntity<AlertRuleResponse> updateRule(
            @PathVariable Long id,
            @RequestBody UpdateAlertRuleRequest request,
            @RequestHeader(value = "X-Operator-Id", defaultValue = "system") String operatorId) {
        try {
            AlertRuleResponse rule = commandService.updateRule(id, request, operatorId);
            return ResponseEntity.ok(rule);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @PutMapping("/{id}/enable")
    public ResponseEntity<Map<String, Object>> enableRule(@PathVariable Long id) {
        try {
            return ResponseEntity.ok(commandService.enableRule(id));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @PutMapping("/{id}/disable")
    public ResponseEntity<Map<String, Object>> disableRule(@PathVariable Long id) {
        try {
            return ResponseEntity.ok(commandService.disableRule(id));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Map<String, Object>> deleteRule(@PathVariable Long id) {
        try {
            return ResponseEntity.ok(commandService.deleteRule(id));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @GetMapping("/check-unique")
    public ResponseEntity<Map<String, Boolean>> checkUnique(
            @RequestParam String field,
            @RequestParam String value,
            @RequestParam(required = false) Long excludeId) {
        boolean isUnique = switch (field) {
            case "ruleName" -> queryService.checkRuleNameUnique(value, excludeId);
            case "ruleCode" -> queryService.checkRuleCodeUnique(value, excludeId);
            default -> false
        };
        return ResponseEntity.ok(Map.of("unique", isUnique));
    }
}