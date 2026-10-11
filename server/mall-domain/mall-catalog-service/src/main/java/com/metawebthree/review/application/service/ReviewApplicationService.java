package com.metawebthree.review.application.service;

import com.metawebthree.review.application.dto.ReviewCreateDTO;
import com.metawebthree.review.application.dto.ReviewDTO;
import com.metawebthree.review.application.dto.ReviewReplyDTO;
import com.metawebthree.review.domain.model.ReviewDO;
import com.metawebthree.review.domain.model.ReviewStatus;
import com.metawebthree.review.domain.repository.ReviewRepository;
import com.metawebthree.review.infrastructure.client.OrderClient;
import com.metawebthree.review.infrastructure.client.ProductClient;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ReviewApplicationService {

    private final ReviewRepository reviewRepository;
    private final OrderClient orderClient;
    private final ProductClient productClient;

    public ReviewApplicationService(ReviewRepository reviewRepository, 
                                     OrderClient orderClient, 
                                     ProductClient productClient) {
        this.reviewRepository = reviewRepository;
        this.orderClient = orderClient;
        this.productClient = productClient;
    }

    /**
     * Create a review
     */
    @Transactional
    public ReviewDTO create(ReviewCreateDTO createDTO, Long userId) {
        // Verify order exists
        var orderDTO = orderClient.getOrderById(createDTO.getOrderId());
        if (orderDTO == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Order not found");
        }

        // Verify user owns the order
        if (orderDTO.getUserId() != userId) {
            throw new BusinessException(ResponseStatus.FORBIDDEN, "Not authorized to review this order");
        }

        // Check if already reviewed
        List<ReviewDO> existingReviews = reviewRepository.findByOrderId(createDTO.getOrderId());
        boolean alreadyReviewed = existingReviews.stream()
                .anyMatch(r -> r.getOrderItemId().equals(createDTO.getOrderItemId()) 
                            && r.getStatus().equals(ReviewStatus.APPROVED.getCode()));
        if (alreadyReviewed) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Already reviewed this item");
        }

        // Create review record
        ReviewDO review = new ReviewDO();
        review.setOrderId(createDTO.getOrderId());
        review.setOrderItemId(createDTO.getOrderItemId());
        review.setProductId(createDTO.getProductId());
        review.setSkuId(createDTO.getSkuId());
        review.setUserId(userId);
        review.setRating(createDTO.getRating());
        review.setContent(createDTO.getContent());
        review.setImages(createDTO.getImages());
        review.setStatus(ReviewStatus.PENDING.getCode());
        review.setLikeCount(0);
        review.setReplyCount(0);
        review.setCreateTime(LocalDateTime.now());
        review.setUpdateTime(LocalDateTime.now());

        reviewRepository.save(review);
        return convertToDTO(review);
    }

    /**
     * Approve or reject review (admin)
     */
    @Transactional
    public ReviewDTO approve(Long reviewId, boolean approved) {
        ReviewDO review = reviewRepository.findById(reviewId);
        if (review == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Review not found");
        }

        if (approved) {
            review.setStatus(ReviewStatus.APPROVED.getCode());
        } else {
            review.setStatus(ReviewStatus.REJECTED.getCode());
        }
        review.setUpdateTime(LocalDateTime.now());

        reviewRepository.save(review);
        return convertToDTO(review);
    }

    /**
     * Reply to review (store)
     */
    @Transactional
    public ReviewDTO reply(ReviewReplyDTO replyDTO) {
        ReviewDO review = reviewRepository.findById(replyDTO.getReviewId());
        if (review == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Review not found");
        }

        if (!review.getStatus().equals(ReviewStatus.APPROVED.getCode())) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Review not approved yet");
        }

        review.setReplyContent(replyDTO.getReplyContent());
        review.setReplyCount(review.getReplyCount() + 1);
        review.setUpdateTime(LocalDateTime.now());

        reviewRepository.save(review);
        return convertToDTO(review);
    }

    /**
     * Like a review
     */
    public ReviewDTO like(Long reviewId) {
        ReviewDO review = reviewRepository.findById(reviewId);
        if (review == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Review not found");
        }

        if (!review.getStatus().equals(ReviewStatus.APPROVED.getCode())) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Review not approved yet");
        }

        reviewRepository.incrementLikeCount(reviewId);
        review = reviewRepository.findById(reviewId);
        return convertToDTO(review);
    }

    /**
     * Get review by ID
     */
    public ReviewDTO getById(Long id) {
        ReviewDO review = reviewRepository.findById(id);
        if (review == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Review not found");
        }
        return convertToDTO(review);
    }

    /**
     * Get reviews by product ID
     */
    public List<ReviewDTO> getByProductId(Long productId) {
        return reviewRepository.findByProductId(productId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get reviews by product ID with sorting
     */
    public List<ReviewDTO> getByProductIdSorted(Long productId, String sortBy) {
        List<ReviewDO> reviews = reviewRepository.findByProductId(productId);
        
        if ("rating_desc".equals(sortBy)) {
            reviews.sort((a, b) -> b.getRating().compareTo(a.getRating()));
        } else if ("rating_asc".equals(sortBy)) {
            reviews.sort((a, b) -> a.getRating().compareTo(b.getRating()));
        } else if ("like_desc".equals(sortBy)) {
            reviews.sort((a, b) -> b.getLikeCount().compareTo(a.getLikeCount()));
        } else if ("time_desc".equals(sortBy)) {
            reviews.sort((a, b) -> b.getCreateTime().compareTo(a.getCreateTime()));
        }

        return reviews.stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get reviews by user ID
     */
    public List<ReviewDTO> getByUserId(Long userId) {
        return reviewRepository.findByUserId(userId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get reviews by store ID
     */
    public List<ReviewDTO> getByStoreId(Long storeId) {
        return reviewRepository.findByStoreId(storeId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get all reviews (admin)
     */
    public List<ReviewDTO> getAll() {
        return reviewRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    private ReviewDTO convertToDTO(ReviewDO review) {
        ReviewDTO dto = new ReviewDTO();
        dto.setId(review.getId());
        dto.setOrderId(review.getOrderId());
        dto.setOrderItemId(review.getOrderItemId());
        dto.setProductId(review.getProductId());
        dto.setSkuId(review.getSkuId());
        dto.setUserId(review.getUserId());
        dto.setStoreId(review.getStoreId());
        dto.setRating(review.getRating());
        dto.setContent(review.getContent());
        dto.setImages(review.getImages());
        dto.setStatus(review.getStatus());
        dto.setLikeCount(review.getLikeCount());
        dto.setReplyCount(review.getReplyCount());
        dto.setReplyContent(review.getReplyContent());
        dto.setCreateTime(review.getCreateTime());
        dto.setUpdateTime(review.getUpdateTime());

        // Set status description
        if (review.getStatus() != null) {
            for (ReviewStatus status : ReviewStatus.values()) {
                if (status.getCode().equals(review.getStatus())) {
                    dto.setStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        return dto;
    }
}