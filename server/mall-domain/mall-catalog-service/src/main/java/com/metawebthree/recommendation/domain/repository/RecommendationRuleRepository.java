package com.metawebthree.recommendation.domain.repository;

import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import java.util.List;
import java.util.Optional;

public interface RecommendationRuleRepository {
    Optional<RecommendationRule> findById(Long id);
    List<RecommendationRule> findByScene(String scene);
    List<RecommendationRule> findByStatus(RecommendationRule.RuleStatus status);
    List<RecommendationRule> findBySceneAndStatus(String scene, RecommendationRule.RuleStatus status);
    List<RecommendationRule> findAll();
    List<RecommendationRule> findActiveRules();
    long count();
    RecommendationRule save(RecommendationRule rule);
    void update(RecommendationRule rule);
    void deleteById(Long id);
}