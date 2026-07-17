package com.metawebthree.dom.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.dom.application.DomApplicationService;
import com.metawebthree.dom.application.dto.SourcingRuleDTO;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/dom/sourcing-rules")
public class SourcingRuleController {

    private final DomApplicationService domApplicationService;

    public SourcingRuleController(DomApplicationService domApplicationService) {
        this.domApplicationService = domApplicationService;
    }

    @RequirePermission(DomPermissions.DOM_RULE_READ)
    @GetMapping
    public List<SourcingRuleDTO> list(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.getSourcingRules();
    }

    @RequirePermission(DomPermissions.DOM_RULE_CREATE)
    @PostMapping
    public SourcingRuleDTO create(
            @Valid @RequestBody SourcingRuleDTO rule,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.createSourcingRule(rule);
    }

    @RequirePermission(DomPermissions.DOM_RULE_UPDATE)
    @PutMapping("/{id}")
    public SourcingRuleDTO update(
            @PathVariable Long id,
            @Valid @RequestBody SourcingRuleDTO rule,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        rule.setId(id);
        return domApplicationService.updateSourcingRule(rule);
    }

    @RequirePermission(DomPermissions.DOM_RULE_DELETE)
    @DeleteMapping("/{id}")
    public void delete(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        domApplicationService.deleteSourcingRule(id);
    }
}
