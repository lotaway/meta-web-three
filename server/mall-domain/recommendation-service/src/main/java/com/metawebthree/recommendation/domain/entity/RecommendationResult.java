package com.metawebthree.recommendation.domain.entity;

import java.time.LocalDateTime;

public class RecommendationResult {

    public enum RecommendationAlgorithm {
        USER_BASED_CF, ITEM_BASED_CF, CONTENT_BASED, POPULARITY, TRENDING, DEEP_LEARNING, HYBRID, CONTEXT_AWARE
    }

    private Long id;
    private Long userId;
    private Long productId;
    private Double score;
    private RecommendationAlgorithm algorithm;
    private String reason;
    private Integer position;
    private LocalDateTime createdAt;
    private LocalDateTime expiresAt;
    private Boolean isClicked;
    private Boolean isPurchased;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public Long getProductId() {
        return productId;
    }

    public void setProductId(Long productId) {
        this.productId = productId;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public RecommendationAlgorithm getAlgorithm() {
        return algorithm;
    }

    public void setAlgorithm(RecommendationAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public Integer getPosition() {
        return position;
    }

    public void setPosition(Integer position) {
        this.position = position;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(LocalDateTime expiresAt) {
        this.expiresAt = expiresAt;
    }

    public Boolean getIsClicked() {
        return isClicked;
    }

    public void setIsClicked(Boolean isClicked) {
        this.isClicked = isClicked;
    }

    public Boolean getIsPurchased() {
        return isPurchased;
    }

    public void setIsPurchased(Boolean isPurchased) {
        this.isPurchased = isPurchased;
    }
}
