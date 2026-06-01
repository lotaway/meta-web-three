package com.metawebthree.product_recommendation.application.dto;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class RecommendationDTO {
    private Long id;
    private Long userId;
    private Long productId;
    private BigDecimal score;
    private String type;
    private String reason;
    private LocalDateTime createdAt;

    public RecommendationDTO() {
    }

    public RecommendationDTO(Long id, Long userId, Long productId, BigDecimal score, 
                             String type, String reason) {
        this.id = id;
        this.userId = userId;
        this.productId = productId;
        this.score = score;
        this.type = type;
        this.reason = reason;
        this.createdAt = LocalDateTime.now();
    }

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

    public BigDecimal getScore() {
        return score;
    }

    public void setScore(BigDecimal score) {
        this.score = score;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}