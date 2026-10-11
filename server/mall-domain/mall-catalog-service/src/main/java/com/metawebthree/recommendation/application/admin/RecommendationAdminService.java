package com.metawebthree.recommendation.application.admin;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class RecommendationAdminService {

    private final RecommendationRuleRepository ruleRepository;
    private final RecommendationRepository recommendationRepository;

    public RecommendationAdminService(RecommendationRuleRepository ruleRepository,
                                      RecommendationRepository recommendationRepository) {
        this.ruleRepository = ruleRepository;
        this.recommendationRepository = recommendationRepository;
    }

    public Page<RecommendationRuleDO> listRules(int pageNum, int pageSize,
                                                  String ruleName, String scene, String status) {
        List<RecommendationRule> all = ruleRepository.findAll().stream()
            .filter(r -> ruleName == null || (r.getRuleName() != null && r.getRuleName().contains(ruleName)))
            .filter(r -> scene == null || (r.getScene() != null && r.getScene().equals(scene)))
            .filter(r -> status == null || (r.getStatus() != null && r.getStatus().name().equals(status)))
            .sorted(Comparator.comparing(RecommendationRule::getPriority).reversed()
                .thenComparing(RecommendationRule::getId))
            .collect(Collectors.toList());

        int total = all.size();
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, total);
        List<RecommendationRule> pageList = start >= total ? List.of() : all.subList(start, end);

        Page<RecommendationRuleDO> page = new Page<>(pageNum, pageSize, total);
        page.setRecords(pageList.stream().map(this::toRuleDO).collect(Collectors.toList()));
        return page;
    }

    public RecommendationRuleDO getRuleById(Long id) {
        return ruleRepository.findById(id).map(this::toRuleDO).orElse(null);
    }

    public RecommendationRuleDO createRule(Map<String, Object> request) {
        RecommendationRule rule = new RecommendationRule();
        rule.setRuleName((String) request.get("ruleName"));
        rule.setScene((String) request.get("scene"));
        rule.setType(RecommendationRule.RuleType.valueOf((String) request.get("type")));
        rule.setStatus(RecommendationRule.RuleStatus.valueOf((String) request.getOrDefault("status", "DRAFT")));
        rule.setPriority((Integer) request.getOrDefault("priority", 0));
        rule.setMaxItems((Integer) request.getOrDefault("maxItems", 10));
        rule.setConditions((String) request.get("conditions"));
        rule.setExclusions((String) request.get("exclusions"));
        rule.setBoostFactor(new BigDecimal(request.getOrDefault("boostFactor", "1.0").toString()));

        return toRuleDO(ruleRepository.save(rule));
    }

    public void updateRule(Long id, Map<String, Object> request) {
        RecommendationRule rule = ruleRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Rule not found: " + id));

        if (request.get("ruleName") != null) rule.setRuleName((String) request.get("ruleName"));
        if (request.get("scene") != null) rule.setScene((String) request.get("scene"));
        if (request.get("priority") != null) rule.setPriority((Integer) request.get("priority"));
        if (request.get("maxItems") != null) rule.setMaxItems((Integer) request.get("maxItems"));
        if (request.get("conditions") != null) rule.setConditions((String) request.get("conditions"));
        if (request.get("exclusions") != null) rule.setExclusions((String) request.get("exclusions"));
        if (request.get("boostFactor") != null)
            rule.setBoostFactor(new BigDecimal(request.get("boostFactor").toString()));

        ruleRepository.update(rule);
    }

    public void deleteRule(Long id) {
        ruleRepository.deleteById(id);
    }

    public void activateRule(Long id) {
        ruleRepository.findById(id).ifPresent(rule -> {
            rule.activate();
            ruleRepository.update(rule);
        });
    }

    public void pauseRule(Long id) {
        ruleRepository.findById(id).ifPresent(rule -> {
            rule.pause();
            ruleRepository.update(rule);
        });
    }

    public void archiveRule(Long id) {
        ruleRepository.findById(id).ifPresent(rule -> {
            rule.archive();
            ruleRepository.update(rule);
        });
    }

    public Page<RecommendationDO> listRecommendations(int pageNum, int pageSize,
                                                         Long userId, String scene) {
        List<Recommendation> all = recommendationRepository.findAll().stream()
            .filter(r -> userId == null || (r.getUserId() != null && r.getUserId().equals(userId)))
            .filter(r -> scene == null || (r.getScene() != null && r.getScene().equals(scene)))
            .sorted(Comparator.comparing(Recommendation::getCreatedAt).reversed())
            .collect(Collectors.toList());

        int total = all.size();
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, total);
        List<Recommendation> pageList = start >= total ? List.of() : all.subList(start, end);

        Page<RecommendationDO> page = new Page<>(pageNum, pageSize, total);
        page.setRecords(pageList.stream().map(this::toRecDO).collect(Collectors.toList()));
        return page;
    }

    public Map<String, Object> getStatistics() {
        long totalRules = ruleRepository.count();
        long activeRules = ruleRepository.findActiveRules().size();
        List<Recommendation> all = recommendationRepository.findAll();
        long totalRecs = all.size();

        long totalClicks = all.stream()
            .mapToLong(r -> r.getClickCount() != null ? r.getClickCount() : 0)
            .sum();
        long totalConversions = all.stream()
            .mapToLong(r -> r.getConversionCount() != null ? r.getConversionCount() : 0)
            .sum();
        long totalImpressions = all.stream()
            .mapToLong(r -> r.getImpressionCount() != null ? r.getImpressionCount() : 0)
            .sum();
        double avgCtr = totalImpressions > 0
            ? Math.round((double) totalClicks / totalImpressions * 10000.0) / 100.0
            : 0.0;
        double avgConversionRate = totalClicks > 0
            ? Math.round((double) totalConversions / totalClicks * 10000.0) / 100.0
            : 0.0;

        Map<String, Long> sceneDistribution = all.stream()
            .filter(r -> r.getScene() != null)
            .collect(Collectors.groupingBy(Recommendation::getScene, Collectors.counting()));
        Map<String, Long> algorithmDistribution = all.stream()
            .filter(r -> r.getAlgorithm() != null)
            .collect(Collectors.groupingBy(
                r -> r.getAlgorithm().name(), Collectors.counting()));

        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalRules", totalRules);
        stats.put("activeRules", activeRules);
        stats.put("totalRecommendations", totalRecs);
        stats.put("totalClicks", totalClicks);
        stats.put("totalConversions", totalConversions);
        stats.put("avgClickThroughRate", avgCtr);
        stats.put("avgConversionRate", avgConversionRate);
        stats.put("sceneDistribution", sceneDistribution);
        stats.put("algorithmDistribution", algorithmDistribution);
        return stats;
    }

    private RecommendationRuleDO toRuleDO(RecommendationRule rule) {
        RecommendationRuleDO doEntity = new RecommendationRuleDO();
        doEntity.setId(rule.getId());
        doEntity.setRuleName(rule.getRuleName());
        doEntity.setScene(rule.getScene());
        doEntity.setType(rule.getType() != null ? rule.getType().name() : null);
        doEntity.setStatus(rule.getStatus() != null ? rule.getStatus().name() : null);
        doEntity.setPriority(rule.getPriority());
        doEntity.setMaxItems(rule.getMaxItems());
        doEntity.setMinScore(rule.getMinScore());
        doEntity.setConditions(rule.getConditions());
        doEntity.setExclusions(rule.getExclusions());
        doEntity.setBoostFactor(rule.getBoostFactor());
        doEntity.setCreatedAt(rule.getCreatedAt());
        doEntity.setUpdatedAt(rule.getUpdatedAt());
        return doEntity;
    }

    private RecommendationDO toRecDO(Recommendation recommendation) {
        RecommendationDO doEntity = new RecommendationDO();
        doEntity.setId(recommendation.getId());
        doEntity.setUserId(recommendation.getUserId());
        doEntity.setScene(recommendation.getScene());
        doEntity.setAlgorithm(recommendation.getAlgorithm() != null ? recommendation.getAlgorithm().name() : null);
        doEntity.setScore(recommendation.getScore());
        doEntity.setStatus(recommendation.getStatus() != null ? recommendation.getStatus().name() : null);
        doEntity.setCreatedAt(recommendation.getCreatedAt());
        doEntity.setExpiresAt(recommendation.getExpiresAt());
        doEntity.setClickCount(recommendation.getClickCount());
        doEntity.setConversionCount(recommendation.getConversionCount());
        doEntity.setImpressionCount(recommendation.getImpressionCount());
        return doEntity;
    }
}
