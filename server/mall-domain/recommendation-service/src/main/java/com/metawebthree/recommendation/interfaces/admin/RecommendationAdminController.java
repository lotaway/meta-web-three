package com.metawebthree.recommendation.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationMapper;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationRuleMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/recommendation")
public class RecommendationAdminController {

    @Autowired
    private RecommendationRuleMapper ruleMapper;

    @Autowired
    private RecommendationMapper recommendationMapper;

    // ==================== Rule Management ====================

    @GetMapping("/rule/list")
    public ApiResponse<Page<RecommendationRuleDO>> listRules(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String ruleName,
            @RequestParam(required = false) String scene,
            @RequestParam(required = false) String status) {
        
        LambdaQueryWrapper<RecommendationRuleDO> wrapper = new LambdaQueryWrapper<RecommendationRuleDO>()
            .like(ruleName != null, RecommendationRuleDO::getRuleName, ruleName)
            .eq(scene != null, RecommendationRuleDO::getScene, scene)
            .eq(status != null, RecommendationRuleDO::getStatus, status)
            .orderByDesc(RecommendationRuleDO::getPriority)
            .orderByAsc(RecommendationRuleDO::getId);

        Page<RecommendationRuleDO> page = new Page<>(pageNum, pageSize);
        Page<RecommendationRuleDO> result = ruleMapper.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/rule/{id}")
    public ApiResponse<RecommendationRuleDO> getRuleById(@PathVariable Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            return ApiResponse.success(rule);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Rule not found");
    }

    @PostMapping("/rule")
    public ApiResponse<RecommendationRuleDO> createRule(@RequestBody Map<String, Object> request) {
        RecommendationRuleDO rule = new RecommendationRuleDO();
        rule.setRuleName((String) request.get("ruleName"));
        rule.setScene((String) request.get("scene"));
        rule.setType((String) request.get("type"));
        rule.setStatus((String) request.getOrDefault("status", "DRAFT"));
        rule.setPriority((Integer) request.getOrDefault("priority", 0));
        rule.setMaxItems((Integer) request.getOrDefault("maxItems", 10));
        rule.setConditions((String) request.get("conditions"));
        rule.setExclusions((String) request.get("exclusions"));
        rule.setBoostFactor(new BigDecimal(request.getOrDefault("boostFactor", "1.0").toString()));
        
        ruleMapper.insert(rule);

        return ApiResponse.success(rule);
    }

    @PutMapping("/rule/{id}")
    public ApiResponse<Void> updateRule(@PathVariable Long id, @RequestBody Map<String, Object> request) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Rule not found");
        }

        if (request.get("ruleName") != null) {
            rule.setRuleName((String) request.get("ruleName"));
        }
        if (request.get("scene") != null) {
            rule.setScene((String) request.get("scene"));
        }
        if (request.get("priority") != null) {
            rule.setPriority((Integer) request.get("priority"));
        }
        if (request.get("maxItems") != null) {
            rule.setMaxItems((Integer) request.get("maxItems"));
        }
        if (request.get("conditions") != null) {
            rule.setConditions((String) request.get("conditions"));
        }
        if (request.get("exclusions") != null) {
            rule.setExclusions((String) request.get("exclusions"));
        }
        if (request.get("boostFactor") != null) {
            rule.setBoostFactor(new BigDecimal(request.get("boostFactor").toString()));
        }

        ruleMapper.updateById(rule);

        return ApiResponse.success();
    }

    @DeleteMapping("/rule/{id}")
    public ApiResponse<Void> deleteRule(@PathVariable Long id) {
        ruleMapper.deleteById(id);
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/activate")
    public ApiResponse<Void> activateRule(@PathVariable Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("ACTIVE");
            ruleMapper.updateById(rule);
        }
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/pause")
    public ApiResponse<Void> pauseRule(@PathVariable Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("PAUSED");
            ruleMapper.updateById(rule);
        }
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/archive")
    public ApiResponse<Void> archiveRule(@PathVariable Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("ARCHIVED");
            ruleMapper.updateById(rule);
        }
        return ApiResponse.success();
    }

    // ==================== Recommendation Records ====================

    @GetMapping("/list")
    public ApiResponse<Page<RecommendationDO>> listRecommendations(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String scene) {
        
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<RecommendationDO>()
            .eq(userId != null, RecommendationDO::getUserId, userId)
            .eq(scene != null, RecommendationDO::getScene, scene)
            .orderByDesc(RecommendationDO::getCreatedAt);

        Page<RecommendationDO> page = new Page<>(pageNum, pageSize);
        Page<RecommendationDO> result = recommendationMapper.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        LambdaQueryWrapper<RecommendationRuleDO> ruleWrapper = new LambdaQueryWrapper<>();
        Long totalRules = ruleMapper.selectCount(ruleWrapper);

        LambdaQueryWrapper<RecommendationRuleDO> activeWrapper = new LambdaQueryWrapper<RecommendationRuleDO>()
            .eq(RecommendationRuleDO::getStatus, "ACTIVE");
        Long activeRules = ruleMapper.selectCount(activeWrapper);

        LambdaQueryWrapper<RecommendationDO> recWrapper = new LambdaQueryWrapper<>();
        Long totalRecs = recommendationMapper.selectCount(recWrapper);

        Map<String, Object> data = Map.of(
            "totalRules", totalRules,
            "activeRules", activeRules,
            "totalRecommendations", totalRecs
        );
        return ApiResponse.success(data);
    }
}