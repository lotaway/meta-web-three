package com.metawebthree.promotion.domain.model;

import java.time.LocalDateTime;
import java.util.List;

public class CouponBatch {
    private String id;
    private Long couponTypeId;
    private Integer totalCount;
    private LocalDateTime createdAt;
    private String merkleRoot;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public Long getCouponTypeId() { return couponTypeId; }
    public void setCouponTypeId(Long couponTypeId) { this.couponTypeId = couponTypeId; }
    public Integer getTotalCount() { return totalCount; }
    public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public String getMerkleRoot() { return merkleRoot; }
    public void setMerkleRoot(String merkleRoot) { this.merkleRoot = merkleRoot; }
}
