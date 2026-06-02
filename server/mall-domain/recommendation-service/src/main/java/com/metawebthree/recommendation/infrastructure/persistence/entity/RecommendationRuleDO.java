package com.metawebthree.recommendation.infrastructure.persistence.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("tb_recommendation_rule")
public class RecommendationRuleDO {

    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    private String ruleName;

    private String scene;

    private String type;

    private String status;

    private Integer priority;

    private Integer maxItems;

    private BigDecimal minScore;

    private String conditions;

    private String exclusions;

    private String targetSkus;

    private BigDecimal boostFactor;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }

    public String getScene() { return scene; }
    public void setScene(String scene) { this.scene = scene; }

    public String getType() { return type; }
    public void setType(String type) { this.type = type; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

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

    public String getTargetSkus() { return targetSkus; }
    public void setTargetSkus(String targetSkus) { this.targetSkus = targetSkus; }

    public BigDecimal getBoostFactor() { return boostFactor; }
    public void setBoostFactor(BigDecimal boostFactor) { this.boostFactor = boostFactor; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}