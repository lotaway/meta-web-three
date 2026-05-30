package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDate;

public class CashPlanLine {
    private Long id;
    private Long cashPlanId;
    private String categoryCode;
    private String categoryName;
    private CashFlowDirection flowDirection;
    private BigDecimal plannedAmount;
    private BigDecimal actualAmount;
    private LocalDate plannedDate;
    private String remark;
    private Integer sort;

    public enum CashFlowDirection {
        INFLOW, OUTFLOW
    }

    public void create(Long cashPlanId, String categoryCode, String categoryName,
                       CashFlowDirection flowDirection, BigDecimal plannedAmount,
                       LocalDate plannedDate, Integer sort) {
        this.cashPlanId = cashPlanId;
        this.categoryCode = categoryCode;
        this.categoryName = categoryName;
        this.flowDirection = flowDirection;
        this.plannedAmount = plannedAmount;
        this.actualAmount = BigDecimal.ZERO;
        this.plannedDate = plannedDate;
        this.sort = sort;
    }

    public void recordActual(BigDecimal amount) {
        this.actualAmount = this.actualAmount.add(amount);
    }

    public BigDecimal getVariance() {
        return plannedAmount.subtract(actualAmount);
    }

    public BigDecimal getVarianceRate() {
        if (plannedAmount.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return getVariance().divide(plannedAmount, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(BigDecimal.valueOf(100));
    }

    // Getters and Setters
    public Long getId() { return id; }
    public Long getCashPlanId() { return cashPlanId; }
    public String getCategoryCode() { return categoryCode; }
    public String getCategoryName() { return categoryName; }
    public CashFlowDirection getFlowDirection() { return flowDirection; }
    public BigDecimal getPlannedAmount() { return plannedAmount; }
    public BigDecimal getActualAmount() { return actualAmount; }
    public LocalDate getPlannedDate() { return plannedDate; }
    public String getRemark() { return remark; }
    public Integer getSort() { return sort; }

    public void setId(Long id) { this.id = id; }
    public void setCashPlanId(Long cashPlanId) { this.cashPlanId = cashPlanId; }
    public void setCategoryCode(String categoryCode) { this.categoryCode = categoryCode; }
    public void setCategoryName(String categoryName) { this.categoryName = categoryName; }
    public void setFlowDirection(CashFlowDirection flowDirection) { this.flowDirection = flowDirection; }
    public void setPlannedAmount(BigDecimal plannedAmount) { this.plannedAmount = plannedAmount; }
    public void setActualAmount(BigDecimal actualAmount) { this.actualAmount = actualAmount; }
    public void setPlannedDate(LocalDate plannedDate) { this.plannedDate = plannedDate; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setSort(Integer sort) { this.sort = sort; }
}