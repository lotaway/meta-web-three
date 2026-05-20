package com.metawebthree.settlement.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class SplitRule {
    private Long id;
    private String ruleNo;
    private String ruleName;
    private SplitType type;
    private Long merchantId;
    private BigDecimal ratio;
    private BigDecimal fixedAmount;
    private BigDecimal minAmount;
    private BigDecimal maxAmount;
    private SplitStatus status;
    private Integer priority;
    private LocalDateTime effectiveDate;
    private LocalDateTime expireDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum SplitType {
        RATIO, FIXED, RATIO_WITH_FLOOR, HYBRID
    }

    public enum SplitStatus {
        ACTIVE, INACTIVE, EXPIRED
    }

    public void create(String ruleNo, String ruleName, SplitType type, Long merchantId) {
        this.ruleNo = ruleNo;
        this.ruleName = ruleName;
        this.type = type;
        this.merchantId = merchantId;
        this.status = SplitStatus.ACTIVE;
        this.priority = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setRatio(BigDecimal ratio) {
        this.ratio = ratio;
        this.updatedAt = LocalDateTime.now();
    }

    public void setFixedAmount(BigDecimal fixedAmount) {
        this.fixedAmount = fixedAmount;
        this.updatedAt = LocalDateTime.now();
    }

    public void setRange(BigDecimal minAmount, BigDecimal maxAmount) {
        this.minAmount = minAmount;
        this.maxAmount = maxAmount;
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal calculateSplitAmount(BigDecimal orderAmount) {
        if (status != SplitStatus.ACTIVE) {
            throw new IllegalStateException("Split rule is not active");
        }
        switch (type) {
            case RATIO:
                return orderAmount.multiply(ratio).divide(BigDecimal.valueOf(100));
            case FIXED:
                return fixedAmount;
            case RATIO_WITH_FLOOR:
                BigDecimal ratioAmount = orderAmount.multiply(ratio).divide(BigDecimal.valueOf(100));
                if (minAmount != null && ratioAmount.compareTo(minAmount) < 0) {
                    return minAmount;
                }
                if (maxAmount != null && ratioAmount.compareTo(maxAmount) > 0) {
                    return maxAmount;
                }
                return ratioAmount;
            case HYBRID:
                BigDecimal hybridAmount = orderAmount.multiply(ratio).divide(BigDecimal.valueOf(100));
                if (fixedAmount != null) {
                    return hybridAmount.add(fixedAmount);
                }
                return hybridAmount;
            default:
                throw new IllegalArgumentException("Unknown split type");
        }
    }

    public void deactivate() {
        status = SplitStatus.INACTIVE;
        updatedAt = LocalDateTime.now();
    }

    public void activate() {
        status = SplitStatus.ACTIVE;
        updatedAt = LocalDateTime.now();
    }

    public void expire() {
        status = SplitStatus.EXPIRED;
        updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getRuleNo() { return ruleNo; }
    public String getRuleName() { return ruleName; }
    public SplitType getType() { return type; }
    public Long getMerchantId() { return merchantId; }
    public BigDecimal getRatio() { return ratio; }
    public BigDecimal getFixedAmount() { return fixedAmount; }
    public BigDecimal getMinAmount() { return minAmount; }
    public BigDecimal getMaxAmount() { return maxAmount; }
    public SplitStatus getStatus() { return status; }
    public Integer getPriority() { return priority; }
    public LocalDateTime getEffectiveDate() { return effectiveDate; }
    public LocalDateTime getExpireDate() { return expireDate; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
}