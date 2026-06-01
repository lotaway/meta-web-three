package com.metawebthree.product_recommendation.domain.model;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

public class ProductProfile {
    private Long id;
    private Long productId;
    private String category;
    private List<String> tags;
    private List<String> attributes;
    private BigDecimal price;
    private BigDecimal averageRating;
    private Integer salesCount;
    private Map<String, Double> embedding;
    private List<Long> similarProductIds;

    public ProductProfile() {
    }

    public ProductProfile(Long productId, String category) {
        this.productId = productId;
        this.category = category;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getProductId() {
        return productId;
    }

    public void setProductId(Long productId) {
        this.productId = productId;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public List<String> getAttributes() {
        return attributes;
    }

    public void setAttributes(List<String> attributes) {
        this.attributes = attributes;
    }

    public BigDecimal getPrice() {
        return price;
    }

    public void setPrice(BigDecimal price) {
        this.price = price;
    }

    public BigDecimal getAverageRating() {
        return averageRating;
    }

    public void setAverageRating(BigDecimal averageRating) {
        this.averageRating = averageRating;
    }

    public Integer getSalesCount() {
        return salesCount;
    }

    public void setSalesCount(Integer salesCount) {
        this.salesCount = salesCount;
    }

    public Map<String, Double> getEmbedding() {
        return embedding;
    }

    public void setEmbedding(Map<String, Double> embedding) {
        this.embedding = embedding;
    }

    public List<Long> getSimilarProductIds() {
        return similarProductIds;
    }

    public void setSimilarProductIds(List<Long> similarProductIds) {
        this.similarProductIds = similarProductIds;
    }
}