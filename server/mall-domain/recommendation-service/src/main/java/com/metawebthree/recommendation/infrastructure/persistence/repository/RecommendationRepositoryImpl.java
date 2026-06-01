package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class RecommendationRepositoryImpl implements RecommendationRepository {

    private final Map<Long, Recommendation> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);
    private final Map<Long, Map<String, Integer>> clickCounts = new ConcurrentHashMap<>();
    private final Map<Long, Map<String, Integer>> conversionCounts = new ConcurrentHashMap<>();

    @Override
    public Optional<Recommendation> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public List<Recommendation> findByUserId(Long userId) {
        return storage.values().stream()
            .filter(r -> r.getUserId().equals(userId))
            .collect(Collectors.toList());
    }

    @Override
    public List<Recommendation> findByUserIdAndScene(Long userId, String scene) {
        return storage.values().stream()
            .filter(r -> r.getUserId().equals(userId))
            .filter(r -> r.getScene().equals(scene))
            .collect(Collectors.toList());
    }

    @Override
    public List<Recommendation> findByStatus(Recommendation.RecommendationStatus status) {
        return storage.values().stream()
            .filter(r -> r.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public Recommendation save(Recommendation recommendation) {
        if (recommendation.getId() == null) {
            recommendation.setId(idGenerator.getAndIncrement());
        }
        storage.put(recommendation.getId(), recommendation);
        return recommendation;
    }

    @Override
    public void update(Recommendation recommendation) {
        if (recommendation.getId() != null && storage.containsKey(recommendation.getId())) {
            storage.put(recommendation.getId(), recommendation);
        }
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
    
    @Override
    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        List<Recommendation> userRecs = findByUserId(userId);
        for (Recommendation rec : userRecs) {
            if (rec.getItems() != null) {
                boolean found = rec.getItems().stream()
                    .anyMatch(item -> item.getSkuCode().equals(skuCode));
                if (found) {
                    incrementMetric(rec.getId(), skuCode, behaviorType);
                    break;
                }
            }
        }
    }
    
    private void incrementMetric(Long recommendationId, String skuCode, String behaviorType) {
        Recommendation rec = storage.get(recommendationId);
        if (rec == null) return;
        
        if ("CLICK".equals(behaviorType)) {
            clickCounts.computeIfAbsent(recommendationId, k -> new ConcurrentHashMap<>())
                .merge(skuCode, 1, Integer::sum);
            rec.setClickCount(clickCounts.get(recommendationId).values().stream()
                .mapToInt(Integer::intValue).sum());
        } else if ("CONVERSION".equals(behaviorType)) {
            conversionCounts.computeIfAbsent(recommendationId, k -> new ConcurrentHashMap<>())
                .merge(skuCode, 1, Integer::sum);
            rec.setConversionCount(conversionCounts.get(recommendationId).values().stream()
                .mapToInt(Integer::intValue).sum());
        } else if ("IMPRESSION".equals(behaviorType)) {
            rec.setImpressionCount((rec.getImpressionCount() != null ? rec.getImpressionCount() : 0) + 1);
        }
    }
}