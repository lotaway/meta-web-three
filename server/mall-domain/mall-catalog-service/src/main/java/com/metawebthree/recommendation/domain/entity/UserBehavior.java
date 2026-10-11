package com.metawebthree.recommendation.domain.entity;

import java.time.LocalDateTime;

public class UserBehavior {

    public enum BehaviorType {
        VIEW, CLICK, CART, PURCHASE, COLLECT, SHARE, REVIEW, SEARCH
    }

    private Long id;
    private Long userId;
    private Long productId;
    private BehaviorType behaviorType;
    private Double behaviorValue;
    private LocalDateTime timestamp;
    private String sessionId;
    private String source;

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

    public Double getBehaviorValue() {
        return behaviorValue;
    }

    public void setBehaviorValue(Double behaviorValue) {
        this.behaviorValue = behaviorValue;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }
}
