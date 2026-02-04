package com.metawebthree.promotion.infrastructure.persistence.model;

import java.time.LocalDateTime;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

@TableName("coupon")
public class CouponRecord {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String code;
    private Long couponTypeId;
    private Long ownerUserId;
    private Integer transferStatus;
    private Integer acquireMethod;
    private Integer useStatus;
    private String orderNo;
    private String consumerName;
    private String operatorName;
    private String batchId;
    private LocalDateTime usedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getCode() { return code; }
    public void setCode(String code) { this.code = code; }
    public Long getCouponTypeId() { return couponTypeId; }
    public void setCouponTypeId(Long couponTypeId) { this.couponTypeId = couponTypeId; }
    public Long getOwnerUserId() { return ownerUserId; }
    public void setOwnerUserId(Long ownerUserId) { this.ownerUserId = ownerUserId; }
    public Integer getTransferStatus() { return transferStatus; }
    public void setTransferStatus(Integer transferStatus) { this.transferStatus = transferStatus; }
    public Integer getAcquireMethod() { return acquireMethod; }
    public void setAcquireMethod(Integer acquireMethod) { this.acquireMethod = acquireMethod; }
    public Integer getUseStatus() { return useStatus; }
    public void setUseStatus(Integer useStatus) { this.useStatus = useStatus; }
    public String getOrderNo() { return orderNo; }
    public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
    public String getConsumerName() { return consumerName; }
    public void setConsumerName(String consumerName) { this.consumerName = consumerName; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }
    public LocalDateTime getUsedAt() { return usedAt; }
    public void setUsedAt(LocalDateTime usedAt) { this.usedAt = usedAt; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
