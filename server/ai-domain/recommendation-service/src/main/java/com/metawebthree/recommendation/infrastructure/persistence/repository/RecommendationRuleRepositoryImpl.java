package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class RecommendationRuleRepositoryImpl implements RecommendationRuleRepository {

    private final Map<Long, RecommendationRule> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public Optional<RecommendationRule> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public List<RecommendationRule> findByScene(String scene) {
        return storage.values().stream()
            .filter(r -> r.getScene().equals(scene))
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationRule> findByStatus(RecommendationRule.RuleStatus status) {
        return storage.values().stream()
            .filter(r -> r.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationRule> findBySceneAndStatus(String scene, 
            RecommendationRule.RuleStatus status) {
        return storage.values().stream()
            .filter(r -> r.getScene().equals(scene))
            .filter(r -> r.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public RecommendationRule save(RecommendationRule rule) {
        if (rule.getId() == null) {
            rule.setId(idGenerator.getAndIncrement());
        }
        storage.put(rule.getId(), rule);
        return rule;
    }

    @Override
    public void update(RecommendationRule rule) {
        if (rule.getId() != null && storage.containsKey(rule.getId())) {
            storage.put(rule.getId(), rule);
        }
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}