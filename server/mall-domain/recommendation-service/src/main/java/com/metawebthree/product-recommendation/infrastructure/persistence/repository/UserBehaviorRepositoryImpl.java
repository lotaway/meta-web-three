package com.metawebthree.product_recommendation.infrastructure.persistence.repository;

import com.metawebthree.product_recommendation.domain.model.UserBehavior;
import com.metawebthree.product_recommendation.domain.repository.UserBehaviorRepository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class UserBehaviorRepositoryImpl implements UserBehaviorRepository {

    private final Map<Long, List<UserBehavior>> storage = new ConcurrentHashMap<>();
    private Long idCounter = 1L;

    @Override
    public void save(UserBehavior behavior) {
        if (behavior.getId() == null) {
            behavior.setId(idCounter++);
        }
        storage.computeIfAbsent(behavior.getUserId(), k -> new ArrayList<>()).add(behavior);
    }

    @Override
    public void batchSave(List<UserBehavior> behaviors) {
        for (UserBehavior behavior : behaviors) {
            save(behavior);
        }
    }

    @Override
    public List<UserBehavior> findByUserId(Long userId) {
        return storage.getOrDefault(userId, new ArrayList<>());
    }

    @Override
    public List<UserBehavior> findByUserIdAfter(Long userId, LocalDateTime after) {
        return storage.getOrDefault(userId, new ArrayList<>()).stream()
                .filter(b -> b.getOccurredAt() != null && b.getOccurredAt().isAfter(after))
                .collect(Collectors.toList());
    }

    @Override
    public List<UserBehavior> findByProductId(Long productId) {
        return storage.values().stream()
                .flatMap(List::stream)
                .filter(b -> productId.equals(b.getProductId()))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteOldRecords(LocalDateTime before) {
        for (List<UserBehavior> behaviors : storage.values()) {
            behaviors.removeIf(b -> b.getOccurredAt() != null && b.getOccurredAt().isBefore(before));
        }
    }
}