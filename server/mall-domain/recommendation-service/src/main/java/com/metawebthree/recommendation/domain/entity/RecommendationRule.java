package com.metawebthree.recommendation.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

public class RecommendationRule {
    private Long id;
    private String ruleName;
    private String scene;
    private RuleType type;
    private RuleStatus status;
    private Integer priority;
    private Integer maxItems;
    private BigDecimal minScore;
    private String conditions;
    private String exclusions;
    private List<String> targetSkus;
    private BigDecimal boostFactor;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum RuleType {
        BOOST, FILTER, RE_RANK, EXCLUDE
    }

    public enum RuleStatus {
        DRAFT, ACTIVE, PAUSED, ARCHIVED
    }

    public void create(String ruleName, String scene, RuleType type) {
        this.ruleName = ruleName;
        this.scene = scene;
        this.type = type;
        this.status = RuleStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = RuleStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void pause() {
        this.status = RuleStatus.PAUSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void archive() {
        this.status = RuleStatus.ARCHIVED;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public String getScene() { return scene; }
    public void setScene(String scene) { this.scene = scene; }
    public RuleType getType() { return type; }
    public void setType(RuleType type) { this.type = type; }
    public RuleStatus getStatus() { return status; }
    public void setStatus(RuleStatus status) { this.status = status; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public Integer getMaxItems() { return maxItems; }
    public void setMaxItems(Integer maxItems) { this.maxItems = maxItems; }
    public BigDecimal getMinScore() { return minScore; }
    public void setMinScore(BigDecimal minScore) { this.minScore = minScore; }
    public String getConditions() { return conditions; }
    public void setConditions(String conditions) { this.conditions = conditions; }
    public String getExclusions() { return exclusions; }
    public void setExclusions(String exclusions) { this.exclusions = exclusions; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<String> getTargetSkus() { return targetSkus; }
    public void setTargetSkus(List<String> targetSkus) { this.targetSkus = targetSkus; }
    public BigDecimal getBoostFactor() { return boostFactor; }
    public void setBoostFactor(BigDecimal boostFactor) { this.boostFactor = boostFactor; }
}