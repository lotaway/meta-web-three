package com.metawebthree.recommendation.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

public class Recommendation {
    private Long id;
    private Long userId;
    private String scene;
    private List<RecommendedItem> items;
    private RecommendationAlgorithm algorithm;
    private BigDecimal score;
    private RecommendationStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime expiresAt;
    private Integer clickCount;
    private Integer conversionCount;
    private Integer impressionCount;

    public enum RecommendationAlgorithm {
        COLLABORATIVE_FILTERING, CONTENT_BASED, HYBRID, POPULARITY, AI_MODEL
    }

    public enum RecommendationStatus {
        GENERATING, COMPLETED, EXPIRED, CLICKED, CONVERTED
    }

    public static class RecommendedItem {
        private String skuCode;
        private String skuName;
        private BigDecimal score;
        private Integer rank;
        private String reason;

        public String getSkuCode() { return skuCode; }
        public void setSkuCode(String skuCode) { this.skuCode = skuCode; }
        public String getSkuName() { return skuName; }
        public void setSkuName(String skuName) { this.skuName = skuName; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public Integer getRank() { return rank; }
        public void setRank(Integer rank) { this.rank = rank; }
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
    }

    public void generate(Long userId, String scene, RecommendationAlgorithm algorithm) {
        this.userId = userId;
        this.scene = scene;
        this.algorithm = algorithm;
        this.status = RecommendationStatus.GENERATING;
        this.createdAt = LocalDateTime.now();
        this.expiresAt = createdAt.plusHours(24);
    }

    public void complete(List<RecommendedItem> items) {
        this.items = items;
        this.status = RecommendationStatus.COMPLETED;
    }

    public void expire() {
        this.status = RecommendationStatus.EXPIRED;
    }

    public void recordClick(String skuCode) {
        if (items != null) {
            items.stream()
                .filter(i -> i.getSkuCode().equals(skuCode))
                .findFirst()
                .ifPresent(item -> this.status = RecommendationStatus.CLICKED);
        }
    }

    public void recordConversion(String skuCode) {
        if (items != null) {
            items.stream()
                .filter(i -> i.getSkuCode().equals(skuCode))
                .findFirst()
                .ifPresent(item -> this.status = RecommendationStatus.CONVERTED);
        }
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getScene() { return scene; }
    public void setScene(String scene) { this.scene = scene; }
    public List<RecommendedItem> getItems() { return items; }
    public void setItems(List<RecommendedItem> items) { this.items = items; }
    public RecommendationAlgorithm getAlgorithm() { return algorithm; }
    public void setAlgorithm(RecommendationAlgorithm algorithm) { this.algorithm = algorithm; }
    public BigDecimal getScore() { return score; }
    public void setScore(BigDecimal score) { this.score = score; }
    public RecommendationStatus getStatus() { return status; }
    public void setStatus(RecommendationStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getExpiresAt() { return expiresAt; }
    public void setExpiresAt(LocalDateTime expiresAt) { this.expiresAt = expiresAt; }
    
    public Integer getClickCount() { return clickCount; }
    public void setClickCount(Integer clickCount) { this.clickCount = clickCount; }
    public Integer getConversionCount() { return conversionCount; }
    public void setConversionCount(Integer conversionCount) { this.conversionCount = conversionCount; }
    public Integer getImpressionCount() { return impressionCount; }
    public void setImpressionCount(Integer impressionCount) { this.impressionCount = impressionCount; }
}