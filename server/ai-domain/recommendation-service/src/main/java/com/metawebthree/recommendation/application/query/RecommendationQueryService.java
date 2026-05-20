package com.metawebthree.recommendation.application.query;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class RecommendationQueryService {

    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;

    public RecommendationQueryService(
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository) {
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
    }

    public Optional<Recommendation> getRecommendationById(Long id) {
        return recommendationRepository.findById(id);
    }

    public List<Recommendation> getUserRecommendations(Long userId) {
        return recommendationRepository.findByUserId(userId);
    }

    public List<Recommendation> getUserRecommendationsByScene(Long userId, String scene) {
        return recommendationRepository.findByUserIdAndScene(userId, scene);
    }

    public List<RecommendationRule> getRulesByScene(String scene) {
        return ruleRepository.findByScene(scene);
    }

    public List<RecommendationRule> getActiveRules(String scene) {
        return ruleRepository.findBySceneAndStatus(scene, RecommendationRule.RuleStatus.ACTIVE);
    }

    public Optional<RecommendationRule> getRuleById(Long id) {
        return ruleRepository.findById(id);
    }
}