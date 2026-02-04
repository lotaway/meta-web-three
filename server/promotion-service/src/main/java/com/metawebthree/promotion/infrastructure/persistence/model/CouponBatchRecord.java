package com.metawebthree.promotion.infrastructure.persistence.model;

import java.time.LocalDateTime;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

@TableName("coupon_batch")
public class CouponBatchRecord {
    @TableId
    private String id;
    private Long couponTypeId;
    private Integer totalCount;
    private LocalDateTime createdAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public Long getCouponTypeId() { return couponTypeId; }
    public void setCouponTypeId(Long couponTypeId) { this.couponTypeId = couponTypeId; }
    public Integer getTotalCount() { return totalCount; }
    public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
