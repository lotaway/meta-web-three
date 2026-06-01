package com.metawebthree.product_recommendation.domain.model;

import java.time.LocalDateTime;

public class UserBehavior {
    private Long id;
    private Long userId;
    private Long productId;
    private BehaviorType behaviorType;
    private Integer durationSeconds;
    private String source;
    private String searchKeyword;
    private LocalDateTime occurredAt;

    public enum BehaviorType {
        VIEW,
        CLICK,
        ADD_TO_CART,
        PURCHASE,
        FAVORITE,
        REVIEW,
        SHARE,
        SEARCH
    }

    public UserBehavior() {
    }

    public UserBehavior(Long id, Long userId, Long productId, BehaviorType behaviorType) {
        this.id = id;
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

    public BehaviorType getBehaviorType() {
        return behaviorType;
    }

    public void setBehaviorType(BehaviorType behaviorType) {
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

    public double getWeight() {
        if (behaviorType == null) {
            return 1.0;
        }
        return switch (behaviorType) {
            case PURCHASE -> 5.0;
            case ADD_TO_CART -> 3.0;
            case FAVORITE -> 2.5;
            case REVIEW, SHARE -> 2.0;
            case CLICK -> 1.5;
            case SEARCH -> 1.2;
            case VIEW -> 1.0;
        };
    }
}