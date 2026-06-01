package com.metawebthree.review.domain.repository;

import com.metawebthree.review.domain.model.ReviewDO;
import java.util.List;

public interface ReviewRepository {
    ReviewDO save(ReviewDO review);
    ReviewDO findById(Long id);
    List<ReviewDO> findByProductId(Long productId);
    List<ReviewDO> findByUserId(Long userId);
    List<ReviewDO> findByStoreId(Long storeId);
    List<ReviewDO> findByOrderId(Long orderId);
    List<ReviewDO> findAll();
    boolean updateStatus(Long id, Integer status);
    boolean deleteById(Long id);
    boolean incrementLikeCount(Long id);
    boolean addReply(Long id, String replyContent);
}