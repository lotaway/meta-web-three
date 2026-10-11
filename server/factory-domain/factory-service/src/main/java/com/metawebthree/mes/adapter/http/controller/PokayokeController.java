package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.application.command.PokayokeRuleService;
import com.metawebthree.mes.infrastructure.persistence.dataobject.PokayokeRuleDO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/mes/pokayoke")
@RequiredArgsConstructor
public class PokayokeController {
    
    private final PokayokeRuleService ruleService;
    
    @PostMapping("/rules")
    public PokayokeRuleDO createRule(@RequestBody PokayokeRuleDO rule) {
        return ruleService.createRule(rule);
    }
    
    @PutMapping("/rules/{id}")
    public PokayokeRuleDO updateRule(@PathVariable Long id, @RequestBody PokayokeRuleDO rule) {
        rule.setId(id);
        return ruleService.updateRule(rule);
    }
    
    @PostMapping("/rules/{id}/activate")
    public void activateRule(@PathVariable Long id) {
        ruleService.activateRule(id);
    }
    
    @PostMapping("/rules/{id}/deactivate")
    public void deactivateRule(@PathVariable Long id) {
        ruleService.deactivateRule(id);
    }
    
    @DeleteMapping("/rules/{id}")
    public void deleteRule(@PathVariable Long id) {
        ruleService.deleteRule(id);
    }
    
    @GetMapping("/rules")
    public List<PokayokeRuleDO> listRules(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String ruleType) {
        return ruleService.listRules(status, ruleType);
    }
    
    @GetMapping("/rules/{id}")
    public PokayokeRuleDO getRule(@PathVariable Long id) {
        return ruleService.getRule(id);
    }
    
    @GetMapping("/rules/active")
    public List<PokayokeRuleDO> getActiveRules() {
        return ruleService.getActiveRules();
    }
    
    @GetMapping("/rules/workstation/{workstationId}")
    public List<PokayokeRuleDO> getRulesByWorkstation(@PathVariable String workstationId) {
        return ruleService.getRulesByWorkstation(workstationId);
    }
}