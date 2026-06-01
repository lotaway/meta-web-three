package com.metawebthree.product_recommendation.infrastructure.persistence.repository;

import com.metawebthree.product_recommendation.domain.model.Recommendation;
import com.metawebthree.product_recommendation.domain.model.RecommendationType;
import com.metawebthree.product_recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.product_recommendation.infrastructure.persistence.mapper.RecommendationMapper;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class ProductRecommendationRepositoryImpl implements RecommendationRepository {

    private final Map<Long, List<Recommendation>> storage = new ConcurrentHashMap<>();
    private Long idCounter = 1L;

    @Override
    public void save(Recommendation recommendation) {
        if (recommendation.getId() == null) {
            recommendation.setId(idCounter++);
        }
        storage.computeIfAbsent(recommendation.getUserId(), k -> new ArrayList<>()).add(recommendation);
    }

    @Override
    public void batchSave(List<Recommendation> recommendations) {
        for (Recommendation recommendation : recommendations) {
            save(recommendation);
        }
    }

    @Override
    public List<Recommendation> findByUserId(Long userId) {
        return storage.getOrDefault(userId, new ArrayList<>());
    }

    @Override
    public List<Recommendation> findByUserIdAndType(Long userId, String type) {
        return storage.getOrDefault(userId, new ArrayList<>()).stream()
                .filter(r -> r.getType() != null && r.getType().name().equals(type))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteByUserId(Long userId) {
        storage.remove(userId);
    }

    @Override
    public void deleteExpired() {
        LocalDateTime now = LocalDateTime.now();
        for (List<Recommendation> recommendations : storage.values()) {
            recommendations.removeIf(r -> r.getExpiresAt() != null && r.getExpiresAt().isBefore(now));
        }
    }
}