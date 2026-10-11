package com.metawebthree.settlement.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class ReconciliationRecord {
    private Long id;
    private String recordNo;
    private ReconciliationType type;
    private LocalDateTime reconcileDate;
    private String channel;
    private BigDecimal totalAmount;
    private Integer totalCount;
    private BigDecimal matchedAmount;
    private Integer matchedCount;
    private BigDecimal unmatchedAmount;
    private Integer unmatchedCount;
    private ReconciliationStatus status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ReconciliationType {
        DAILY, MONTHLY, MANUAL
    }

    public enum ReconciliationStatus {
        PROCESSING, COMPLETED, FAILED
    }

    public void create(String recordNo, ReconciliationType type, String channel, 
                       BigDecimal totalAmount, Integer totalCount) {
        this.recordNo = recordNo;
        this.type = type;
        this.channel = channel;
        this.totalAmount = totalAmount;
        this.totalCount = totalCount;
        this.matchedAmount = BigDecimal.ZERO;
        this.matchedCount = 0;
        this.unmatchedAmount = BigDecimal.ZERO;
        this.unmatchedCount = 0;
        this.status = ReconciliationStatus.PROCESSING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void recordMatch(BigDecimal amount) {
        matchedAmount = matchedAmount.add(amount);
        matchedCount++;
        unmatchedAmount = unmatchedAmount.subtract(amount);
        unmatchedCount--;
        updatedAt = LocalDateTime.now();
    }

    public void recordMismatch(BigDecimal amount) {
        unmatchedAmount = unmatchedAmount.add(amount);
        unmatchedCount++;
        updatedAt = LocalDateTime.now();
    }

    public void complete() {
        if (status != ReconciliationStatus.PROCESSING) {
            throw new IllegalStateException("Can only complete processing records");
        }
        status = ReconciliationStatus.COMPLETED;
        reconcileDate = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    public void fail(String remark) {
        if (status != ReconciliationStatus.PROCESSING) {
            throw new IllegalStateException("Can only fail processing records");
        }
        status = ReconciliationStatus.FAILED;
        this.remark = remark;
        updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getRecordNo() { return recordNo; }
    public ReconciliationType getType() { return type; }
    public LocalDateTime getReconcileDate() { return reconcileDate; }
    public String getChannel() { return channel; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public Integer getTotalCount() { return totalCount; }
    public BigDecimal getMatchedAmount() { return matchedAmount; }
    public Integer getMatchedCount() { return matchedCount; }
    public BigDecimal getUnmatchedAmount() { return unmatchedAmount; }
    public Integer getUnmatchedCount() { return unmatchedCount; }
    public ReconciliationStatus getStatus() { return status; }
    public String getRemark() { return remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
}