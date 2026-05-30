package com.metawebthree.finance.domain.entity.budget;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class Budget {
    private Long id;
    private String budgetCode;
    private String budgetName;
    private BudgetType type;
    private BudgetPeriod period;
    private Long departmentId;
    private String departmentName;
    private BudgetStatus status;
    private BigDecimal totalAmount;
    private BigDecimal usedAmount;
    private BigDecimal adjustedAmount;
    private String currency;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime approvedAt;
    private Long approvedBy;
    private String approverName;
    private String remark;
    private List<BudgetLine> lines;

    public enum BudgetType {
        ANNUAL, QUARTERLY, MONTHLY, PROJECT
    }

    public enum BudgetPeriod {
        FY2025, FY2026, FY2027, Q1_2025, Q2_2025, Q3_2025, Q4_2025, Q1_2026, Q2_2026
    }

    public enum BudgetStatus {
        DRAFT, PENDING_APPROVAL, APPROVED, REJECTED, CLOSED
    }

    public void create(String budgetCode, String budgetName, BudgetType type, BudgetPeriod period,
                       Long departmentId, String departmentName, Long createdBy, String creatorName) {
        this.budgetCode = budgetCode;
        this.budgetName = budgetName;
        this.type = type;
        this.period = period;
        this.departmentId = departmentId;
        this.departmentName = departmentName;
        this.status = BudgetStatus.DRAFT;
        this.totalAmount = BigDecimal.ZERO;
        this.usedAmount = BigDecimal.ZERO;
        this.adjustedAmount = BigDecimal.ZERO;
        this.currency = "CNY";
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.lines = new ArrayList<>();
    }

    public void submitForApproval() {
        if (status != BudgetStatus.DRAFT) {
            return;
        }
        if (lines == null || lines.isEmpty()) {
            return;
        }
        status = BudgetStatus.PENDING_APPROVAL;
        updatedAt = LocalDateTime.now();
    }

    public void approve(Long approvedBy, String approverName) {
        if (status != BudgetStatus.PENDING_APPROVAL) {
            return;
        }
        status = BudgetStatus.APPROVED;
        this.approvedBy = approvedBy;
        this.approverName = approverName;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reject() {
        if (status != BudgetStatus.PENDING_APPROVAL) {
            return;
        }
        status = BudgetStatus.REJECTED;
        updatedAt = LocalDateTime.now();
    }

    public void close() {
        if (status != BudgetStatus.APPROVED) {
            return;
        }
        status = BudgetStatus.CLOSED;
        updatedAt = LocalDateTime.now();
    }

    public void addLine(BudgetLine line) {
        if (lines == null) {
            lines = new ArrayList<>();
        }
        lines.add(line);
        recalculateTotal();
        updatedAt = LocalDateTime.now();
    }

    public void removeLine(Long lineId) {
        if (lines == null) {
            return;
        }
        lines.removeIf(line -> line.getId().equals(lineId));
        recalculateTotal();
        updatedAt = LocalDateTime.now();
    }

    private void recalculateTotal() {
        if (lines == null || lines.isEmpty()) {
            totalAmount = BigDecimal.ZERO;
            return;
        }
        totalAmount = lines.stream()
                .map(BudgetLine::getBudgetAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    public void applyAdjustment(BigDecimal amount, String reason) {
        adjustedAmount = adjustedAmount.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public void recordUsage(BigDecimal amount) {
        usedAmount = usedAmount.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getAvailableAmount() {
        return totalAmount.add(adjustedAmount).subtract(usedAmount);
    }

    public BigDecimal getUsageRate() {
        BigDecimal base = totalAmount.add(adjustedAmount);
        if (base.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return usedAmount.divide(base, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(BigDecimal.valueOf(100));
    }

    public Long getId() { return id; }
    public String getBudgetCode() { return budgetCode; }
    public String getBudgetName() { return budgetName; }
    public BudgetType getType() { return type; }
    public BudgetPeriod getPeriod() { return period; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public BudgetStatus getStatus() { return status; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public BigDecimal getUsedAmount() { return usedAmount; }
    public BigDecimal getAdjustedAmount() { return adjustedAmount; }
    public String getCurrency() { return currency; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public Long getApprovedBy() { return approvedBy; }
    public String getApproverName() { return approverName; }
    public String getRemark() { return remark; }
    public List<BudgetLine> getLines() { return lines; }

    public void setId(Long id) { this.id = id; }
    public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
    public void setBudgetName(String budgetName) { this.budgetName = budgetName; }
    public void setType(BudgetType type) { this.type = type; }
    public void setPeriod(BudgetPeriod period) { this.period = period; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setStatus(BudgetStatus status) { this.status = status; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public void setUsedAmount(BigDecimal usedAmount) { this.usedAmount = usedAmount; }
    public void setAdjustedAmount(BigDecimal adjustedAmount) { this.adjustedAmount = adjustedAmount; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public void setApprovedBy(Long approvedBy) { this.approvedBy = approvedBy; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setLines(List<BudgetLine> lines) { this.lines = lines; }
}