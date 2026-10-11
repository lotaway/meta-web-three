package com.metawebthree.recommendation.infrastructure.persistence.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.LocalDateTime;

@TableName("tb_product_similarity")
public class ProductSimilarityDO {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long productId1;
    private Long productId2;
    private Double similarityScore;
    private String algorithm;
    private LocalDateTime lastUpdated;
    private Integer updateCount;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getProductId1() {
        return productId1;
    }

    public void setProductId1(Long productId1) {
        this.productId1 = productId1;
    }

    public Long getProductId2() {
        return productId2;
    }

    public void setProductId2(Long productId2) {
        this.productId2 = productId2;
    }

    public Double getSimilarityScore() {
        return similarityScore;
    }

    public void setSimilarityScore(Double similarityScore) {
        this.similarityScore = similarityScore;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    public LocalDateTime getLastUpdated() {
        return lastUpdated;
    }

    public void setLastUpdated(LocalDateTime lastUpdated) {
        this.lastUpdated = lastUpdated;
    }

    public Integer getUpdateCount() {
        return updateCount;
    }

    public void setUpdateCount(Integer updateCount) {
        this.updateCount = updateCount;
    }
}
