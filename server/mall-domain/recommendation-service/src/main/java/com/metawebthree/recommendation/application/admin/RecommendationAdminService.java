package com.metawebthree.recommendation.application.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationMapper;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationRuleMapper;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.Map;

@Service
public class RecommendationAdminService {

    private final RecommendationRuleMapper ruleMapper;
    private final RecommendationMapper recommendationMapper;

    public RecommendationAdminService(RecommendationRuleMapper ruleMapper,
                                      RecommendationMapper recommendationMapper) {
        this.ruleMapper = ruleMapper;
        this.recommendationMapper = recommendationMapper;
    }

    public Page<RecommendationRuleDO> listRules(int pageNum, int pageSize,
                                                  String ruleName, String scene, String status) {
        LambdaQueryWrapper<RecommendationRuleDO> wrapper = new LambdaQueryWrapper<RecommendationRuleDO>()
            .like(ruleName != null, RecommendationRuleDO::getRuleName, ruleName)
            .eq(scene != null, RecommendationRuleDO::getScene, scene)
            .eq(status != null, RecommendationRuleDO::getStatus, status)
            .orderByDesc(RecommendationRuleDO::getPriority)
            .orderByAsc(RecommendationRuleDO::getId);

        Page<RecommendationRuleDO> page = new Page<>(pageNum, pageSize);
        return ruleMapper.selectPage(page, wrapper);
    }

    public RecommendationRuleDO getRuleById(Long id) {
        return ruleMapper.selectById(id);
    }

    public RecommendationRuleDO createRule(Map<String, Object> request) {
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
        return rule;
    }

    public void updateRule(Long id, Map<String, Object> request) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule == null) {
            throw new IllegalArgumentException("Rule not found: " + id);
        }

        if (request.get("ruleName") != null) rule.setRuleName((String) request.get("ruleName"));
        if (request.get("scene") != null) rule.setScene((String) request.get("scene"));
        if (request.get("priority") != null) rule.setPriority((Integer) request.get("priority"));
        if (request.get("maxItems") != null) rule.setMaxItems((Integer) request.get("maxItems"));
        if (request.get("conditions") != null) rule.setConditions((String) request.get("conditions"));
        if (request.get("exclusions") != null) rule.setExclusions((String) request.get("exclusions"));
        if (request.get("boostFactor") != null)
            rule.setBoostFactor(new BigDecimal(request.get("boostFactor").toString()));

        ruleMapper.updateById(rule);
    }

    public void deleteRule(Long id) {
        ruleMapper.deleteById(id);
    }

    public void activateRule(Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("ACTIVE");
            ruleMapper.updateById(rule);
        }
    }

    public void pauseRule(Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("PAUSED");
            ruleMapper.updateById(rule);
        }
    }

    public void archiveRule(Long id) {
        RecommendationRuleDO rule = ruleMapper.selectById(id);
        if (rule != null) {
            rule.setStatus("ARCHIVED");
            ruleMapper.updateById(rule);
        }
    }

    public Page<RecommendationDO> listRecommendations(int pageNum, int pageSize,
                                                        Long userId, String scene) {
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<RecommendationDO>()
            .eq(userId != null, RecommendationDO::getUserId, userId)
            .eq(scene != null, RecommendationDO::getScene, scene)
            .orderByDesc(RecommendationDO::getCreatedAt);

        Page<RecommendationDO> page = new Page<>(pageNum, pageSize);
        return recommendationMapper.selectPage(page, wrapper);
    }

    public Map<String, Object> getStatistics() {
        Long totalRules = ruleMapper.selectCount(new LambdaQueryWrapper<>());
        Long activeRules = ruleMapper.selectCount(
            new LambdaQueryWrapper<RecommendationRuleDO>()
                .eq(RecommendationRuleDO::getStatus, "ACTIVE"));
        Long totalRecs = recommendationMapper.selectCount(new LambdaQueryWrapper<>());

        return Map.of(
            "totalRules", totalRules,
            "activeRules", activeRules,
            "totalRecommendations", totalRecs
        );
    }
}
