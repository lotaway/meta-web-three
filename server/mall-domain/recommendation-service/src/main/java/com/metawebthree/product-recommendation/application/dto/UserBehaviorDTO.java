package com.metawebthree.product_recommendation.application.dto;

import java.time.LocalDateTime;

public class UserBehaviorDTO {
    private Long id;
    private Long userId;
    private Long productId;
    private String behaviorType;
    private Integer durationSeconds;
    private String source;
    private String searchKeyword;
    private LocalDateTime occurredAt;

    public UserBehaviorDTO() {
    }

    public UserBehaviorDTO(Long userId, Long productId, String behaviorType) {
        this.userId = userId;
        this.productId = productId;
        this.behaviorType = behaviorType;
        this.occurredAt = LocalDateTime.now();
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

    public String getBehaviorType() {
        return behaviorType;
    }

    public void setBehaviorType(String behaviorType) {
        this.behaviorType = behaviorType;
    }

    public Integer getDurationSeconds() {
        return durationSeconds;
    }

    public void setDurationSeconds(Integer durationSeconds) {
        this.durationSeconds = durationSeconds;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getSearchKeyword() {
        return searchKeyword;
    }

    public void setSearchKeyword(String searchKeyword) {
        this.searchKeyword = searchKeyword;
    }

    public LocalDateTime getOccurredAt() {
        return occurredAt;
    }

    public void setOccurredAt(LocalDateTime occurredAt) {
        this.occurredAt = occurredAt;
    }
}