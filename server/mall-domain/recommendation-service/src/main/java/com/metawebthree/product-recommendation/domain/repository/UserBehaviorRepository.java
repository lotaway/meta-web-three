package com.metawebthree.product_recommendation.domain.repository;

import com.metawebthree.product_recommendation.domain.model.UserBehavior;
import java.time.LocalDateTime;
import java.util.List;

public interface UserBehaviorRepository {
    void save(UserBehavior behavior);
    void batchSave(List<UserBehavior> behaviors);
    List<UserBehavior> findByUserId(Long userId);
    List<UserBehavior> findByUserIdAfter(Long userId, LocalDateTime after);
    List<UserBehavior> findByProductId(Long productId);
    void deleteOldRecords(LocalDateTime before);
}