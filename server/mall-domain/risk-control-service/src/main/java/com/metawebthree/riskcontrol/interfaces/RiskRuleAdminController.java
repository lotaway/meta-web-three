package com.metawebthree.riskcontrol.interfaces;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.riskcontrol.domain.RiskRule;
import com.metawebthree.riskcontrol.repository.RiskRuleRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/admin/risk/rule")
public class RiskRuleAdminController {

    @Autowired
    private RiskRuleRepository riskRuleRepository;

    @GetMapping("/list")
    public ApiResponse<Page<RiskRule>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String ruleName,
            @RequestParam(required = false) String scene,
            @RequestParam(required = false) String ruleType,
            @RequestParam(required = false) Integer status) {
        
        LambdaQueryWrapper<RiskRule> wrapper = new LambdaQueryWrapper<RiskRule>()
            .eq(RiskRule::getDeleted, 0)
            .like(ruleName != null, RiskRule::getRuleName, ruleName)
            .eq(scene != null, RiskRule::getScene, scene)
            .eq(ruleType != null, RiskRule::getRuleType, ruleType)
            .eq(status != null, RiskRule::getStatus, status)
            .orderByDesc(RiskRule::getPriority)
            .orderByAsc(RiskRule::getId);

        Page<RiskRule> page = new Page<>(pageNum, pageSize);
        Page<RiskRule> result = riskRuleRepository.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/{id}")
    public ApiResponse<RiskRule> getById(@PathVariable Long id) {
        RiskRule rule = riskRuleRepository.selectById(id);
        if (rule != null && rule.getDeleted() == 0) {
            return ApiResponse.success(rule);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Risk rule not found");
    }

    @PostMapping
    public ApiResponse<RiskRule> create(@RequestBody RiskRule rule) {
        rule.setId(null);
        rule.setCreateTime(System.currentTimeMillis());
        rule.setUpdateTime(System.currentTimeMillis());
        rule.setDeleted(0);
        if (rule.getStatus() == null) {
            rule.setStatus(0);
        }
        riskRuleRepository.insert(rule);

        return ApiResponse.success(rule);
    }

    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody RiskRule rule) {
        rule.setId(id);
        rule.setUpdateTime(System.currentTimeMillis());
        riskRuleRepository.updateById(rule);

        return ApiResponse.success();
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        RiskRule rule = new RiskRule();
        rule.setId(id);
        rule.setDeleted(1);
        rule.setUpdateTime(System.currentTimeMillis());
        riskRuleRepository.updateById(rule);

        return ApiResponse.success();
    }

    @PutMapping("/{id}/status")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        RiskRule rule = new RiskRule();
        rule.setId(id);
        rule.setStatus(status);
        rule.setUpdateTime(System.currentTimeMillis());
        riskRuleRepository.updateById(rule);

        return ApiResponse.success();
    }
}