package com.metawebthree.recommendation.domain.repository;

import com.metawebthree.recommendation.domain.entity.UserBehavior;
import java.time.LocalDateTime;
import java.util.List;

public interface UserBehaviorRepository {
    UserBehavior save(UserBehavior userBehavior);
    List<UserBehavior> findByUserIdOrderByTimestampDesc(Long userId);
    List<UserBehavior> findByProductIdOrderByTimestampDesc(Long productId);
    List<UserBehavior> findByUserIdAndBehaviorTypeOrderByTimestampDesc(Long userId, UserBehavior.BehaviorType behaviorType);
    List<UserBehavior> findRecentByUserId(Long userId, LocalDateTime since);
    List<Long> findUserIdsByProductId(Long productId);
    Long countBehavior(Long userId, Long productId, UserBehavior.BehaviorType behaviorType);
    void deleteByTimestampBefore(LocalDateTime timestamp);
    List<UserBehavior> findAll();
    void deleteById(Long id);
}
