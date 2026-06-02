package com.metawebthree.recommendation.infrastructure.persistence.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("tb_recommendation")
public class RecommendationDO {

    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String scene;

    private String algorithm;

    private BigDecimal score;

    private String status;

    private LocalDateTime createdAt;

    private LocalDateTime expiresAt;

    private Integer clickCount;

    private Integer conversionCount;

    private Integer impressionCount;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }

    public String getScene() { return scene; }
    public void setScene(String scene) { this.scene = scene; }

    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }

    public BigDecimal getScore() { return score; }
    public void setScore(BigDecimal score) { this.score = score; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getExpiresAt() { return expiresAt; }
    public void setExpiresAt(LocalDateTime expiresAt) { this.expiresAt = expiresAt; }

    public Integer getClickCount() { return clickCount; }
    public void setClickCount(Integer clickCount) { this.clickCount = clickCount; }

    public Integer getConversionCount() { return conversionCount; }
    public void setConversionCount(Integer conversionCount) { this.conversionCount = conversionCount; }

    public Integer getImpressionCount() { return impressionCount; }
    public void setImpressionCount(Integer impressionCount) { this.impressionCount = impressionCount; }
}