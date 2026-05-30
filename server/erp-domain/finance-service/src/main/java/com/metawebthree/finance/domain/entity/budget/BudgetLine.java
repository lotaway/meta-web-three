package com.metawebthree.finance.domain.entity.budget;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class BudgetLine {
    private Long id;
    private Long budgetId;
    private String subjectCode;
    private String subjectName;
    private BigDecimal budgetAmount;
    private BigDecimal usedAmount;
    private BigDecimal adjustedAmount;
    private Integer sort;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long budgetId, String subjectCode, String subjectName, 
                       BigDecimal budgetAmount, Integer sort) {
        this.budgetId = budgetId;
        this.subjectCode = subjectCode;
        this.subjectName = subjectName;
        this.budgetAmount = budgetAmount;
        this.usedAmount = BigDecimal.ZERO;
        this.adjustedAmount = BigDecimal.ZERO;
        this.sort = sort;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void adjustAmount(BigDecimal amount) {
        adjustedAmount = adjustedAmount.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public void recordUsage(BigDecimal amount) {
        usedAmount = usedAmount.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getAvailableAmount() {
        return budgetAmount.add(adjustedAmount).subtract(usedAmount);
    }

    public BigDecimal getUsageRate() {
        BigDecimal base = budgetAmount.add(adjustedAmount);
        if (base.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return usedAmount.divide(base, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(BigDecimal.valueOf(100));
    }

    public Long getId() { return id; }
    public Long getBudgetId() { return budgetId; }
    public String getSubjectCode() { return subjectCode; }
    public String getSubjectName() { return subjectName; }
    public BigDecimal getBudgetAmount() { return budgetAmount; }
    public BigDecimal getUsedAmount() { return usedAmount; }
    public BigDecimal getAdjustedAmount() { return adjustedAmount; }
    public Integer getSort() { return sort; }
    public String getRemark() { return remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setBudgetId(Long budgetId) { this.budgetId = budgetId; }
    public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
    public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
    public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
    public void setUsedAmount(BigDecimal usedAmount) { this.usedAmount = usedAmount; }
    public void setAdjustedAmount(BigDecimal adjustedAmount) { this.adjustedAmount = adjustedAmount; }
    public void setSort(Integer sort) { this.sort = sort; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}