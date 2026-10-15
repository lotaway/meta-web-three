package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import com.metawebthree.mes.domain.repository.WorkOrderSplitRuleRepository;
import com.metawebthree.mes.domain.service.WorkOrderSplitService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import java.util.List;

@RestController
@RequestMapping("/api/mes/work-order-splits")
public class WorkOrderSplitController {
    private final WorkOrderSplitRuleRepository rules;
    private final WorkOrderSplitService service;
    public WorkOrderSplitController(WorkOrderSplitRuleRepository rules, WorkOrderSplitService service) {
        this.rules = rules;
        this.service = service;
    }

    public record RuleRequest(@NotBlank String ruleCode, @NotBlank String ruleName,
            @NotNull @Pattern(regexp = "BY_QUANTITY|MANUAL") String splitType,
            @Positive Integer splitQuantity, @Min(2) @Max(1000) Integer maxChildOrders) {}
    public record SplitRequest(@NotNull @Positive Long ruleId,
            @Size(min = 2, max = 1000) List<@NotNull @Positive Integer> quantities) {}

    @PostMapping("/rules")
    public WorkOrderSplitRule createRule(@Valid @RequestBody RuleRequest request) {
        if ("BY_QUANTITY".equals(request.splitType()) && request.splitQuantity() == null)
            throw new IllegalArgumentException("BY_QUANTITY requires splitQuantity");
        WorkOrderSplitRule rule = new WorkOrderSplitRule();
        rule.create(request.ruleName(), request.ruleCode(), request.splitType());
        rule.setSplitQuantity(request.splitQuantity());
        rule.setMaxChildOrders(request.maxChildOrders());
        return rules.save(rule);
    }

    @GetMapping("/rules")
    public List<WorkOrderSplitRule> listRules() { return rules.findAll(); }

    @GetMapping("/rules/{id}")
    public WorkOrderSplitRule rule(@PathVariable Long id) {
        return rules.findById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
    }

    @PostMapping("/{parentId}")
    public List<WorkOrder> split(@PathVariable Long parentId, @Valid @RequestBody SplitRequest request) {
        WorkOrder parent = new WorkOrder();
        parent.setId(parentId);
        WorkOrderSplitRule rule = rule(request.ruleId());
        if ("MANUAL".equals(rule.getSplitType())) return service.splitManually(parent, rule, request.quantities());
        if (request.quantities() != null) throw new IllegalArgumentException("Batch split does not accept quantities");
        return service.splitWorkOrder(parent, request.ruleId());
    }

    @GetMapping("/{parentId}/children")
    public List<WorkOrder> children(@PathVariable Long parentId) { return service.getChildWorkOrders(parentId); }
}
