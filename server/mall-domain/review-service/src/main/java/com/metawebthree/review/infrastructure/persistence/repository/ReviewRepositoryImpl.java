package com.metawebthree.review.infrastructure.persistence.repository;

import com.metawebthree.review.domain.model.ReviewDO;
import com.metawebthree.review.domain.repository.ReviewRepository;
import com.metawebthree.review.infrastructure.persistence.mapper.ReviewMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class ReviewRepositoryImpl implements ReviewRepository {

    private final ReviewMapper reviewMapper;

    public ReviewRepositoryImpl(ReviewMapper reviewMapper) {
        this.reviewMapper = reviewMapper;
    }

    @Override
    public ReviewDO save(ReviewDO review) {
        if (review.getId() == null) {
            reviewMapper.insert(review);
        } else {
            reviewMapper.update(review);
        }
        return review;
    }

    @Override
    public ReviewDO findById(Long id) {
        return reviewMapper.selectById(id);
    }

    @Override
    public List<ReviewDO> findByProductId(Long productId) {
        return reviewMapper.selectByProductId(productId);
    }

    @Override
    public List<ReviewDO> findByUserId(Long userId) {
        return reviewMapper.selectByUserId(userId);
    }

    @Override
    public List<ReviewDO> findByStoreId(Long storeId) {
        return reviewMapper.selectByStoreId(storeId);
    }

    @Override
    public List<ReviewDO> findByOrderId(Long orderId) {
        return reviewMapper.selectByOrderId(orderId);
    }

    @Override
    public List<ReviewDO> findAll() {
        return reviewMapper.selectAll();
    }

    @Override
    public boolean updateStatus(Long id, Integer status) {
        return reviewMapper.updateStatus(id, status) > 0;
    }

    @Override
    public boolean deleteById(Long id) {
        return reviewMapper.deleteById(id) > 0;
    }

    @Override
    public boolean incrementLikeCount(Long id) {
        return reviewMapper.incrementLikeCount(id) > 0;
    }

    @Override
    public boolean addReply(Long id, String replyContent) {
        return reviewMapper.addReply(id, replyContent) > 0;
    }
}