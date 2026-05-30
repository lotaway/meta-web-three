package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class CashPlan {
    private Long id;
    private String planCode;
    private String planName;
    private CashPlanType type;
    private CashPlanPeriod period;
    private LocalDate startDate;
    private LocalDate endDate;
    private CashPlanStatus status;
    private BigDecimal totalAmount;
    private BigDecimal inflowAmount;
    private BigDecimal outflowAmount;
    private String currency;
    private Long departmentId;
    private String departmentName;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime approvedAt;
    private Long approvedBy;
    private String approverName;
    private String remark;
    private List<CashPlanLine> lines;

    public enum CashPlanType {
        ANNUAL, QUARTERLY, MONTHLY, WEEKLY
    }

    public enum CashPlanPeriod {
        FY2025, FY2026, FY2027, Q1_2025, Q2_2025, Q3_2025, Q4_2025, Q1_2026, Q2_2026
    }

    public enum CashPlanStatus {
        DRAFT, PENDING_APPROVAL, APPROVED, REJECTED, CLOSED
    }

    public void create(String planCode, String planName, CashPlanType type, CashPlanPeriod period,
                       LocalDate startDate, LocalDate endDate, Long departmentId, String departmentName,
                       Long createdBy, String creatorName) {
        this.planCode = planCode;
        this.planName = planName;
        this.type = type;
        this.period = period;
        this.startDate = startDate;
        this.endDate = endDate;
        this.status = CashPlanStatus.DRAFT;
        this.totalAmount = BigDecimal.ZERO;
        this.inflowAmount = BigDecimal.ZERO;
        this.outflowAmount = BigDecimal.ZERO;
        this.currency = "CNY";
        this.departmentId = departmentId;
        this.departmentName = departmentName;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.lines = new ArrayList<>();
    }

    public void submitForApproval() {
        if (status != CashPlanStatus.DRAFT) {
            return;
        }
        if (lines == null || lines.isEmpty()) {
            return;
        }
        status = CashPlanStatus.PENDING_APPROVAL;
        updatedAt = LocalDateTime.now();
    }

    public void approve(Long approvedBy, String approverName) {
        if (status != CashPlanStatus.PENDING_APPROVAL) {
            return;
        }
        status = CashPlanStatus.APPROVED;
        this.approvedBy = approvedBy;
        this.approverName = approverName;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reject() {
        if (status != CashPlanStatus.PENDING_APPROVAL) {
            return;
        }
        status = CashPlanStatus.REJECTED;
        updatedAt = LocalDateTime.now();
    }

    public void close() {
        if (status != CashPlanStatus.APPROVED) {
            return;
        }
        status = CashPlanStatus.CLOSED;
        updatedAt = LocalDateTime.now();
    }

    public void addLine(CashPlanLine line) {
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
            inflowAmount = BigDecimal.ZERO;
            outflowAmount = BigDecimal.ZERO;
            return;
        }
        inflowAmount = lines.stream()
                .filter(line -> line.getFlowDirection() == CashPlanLine.CashFlowDirection.INFLOW)
                .map(CashPlanLine::getPlannedAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        outflowAmount = lines.stream()
                .filter(line -> line.getFlowDirection() == CashPlanLine.CashFlowDirection.OUTFLOW)
                .map(CashPlanLine::getPlannedAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        totalAmount = inflowAmount.subtract(outflowAmount);
    }

    public void recordActual(BigDecimal amount, CashPlanLine.CashFlowDirection direction) {
        if (direction == CashPlanLine.CashFlowDirection.INFLOW) {
            inflowAmount = inflowAmount.add(amount);
        } else {
            outflowAmount = outflowAmount.add(amount);
        }
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getNetCashFlow() {
        return inflowAmount.subtract(outflowAmount);
    }

    public BigDecimal getActualAmount() {
        return inflowAmount.subtract(outflowAmount);
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getPlanCode() { return planCode; }
    public String getPlanName() { return planName; }
    public CashPlanType getType() { return type; }
    public CashPlanPeriod getPeriod() { return period; }
    public LocalDate getStartDate() { return startDate; }
    public LocalDate getEndDate() { return endDate; }
    public CashPlanStatus getStatus() { return status; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public BigDecimal getInflowAmount() { return inflowAmount; }
    public BigDecimal getOutflowAmount() { return outflowAmount; }
    public String getCurrency() { return currency; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public Long getApprovedBy() { return approvedBy; }
    public String getApproverName() { return approverName; }
    public String getRemark() { return remark; }
    public List<CashPlanLine> getLines() { return lines; }

    public void setId(Long id) { this.id = id; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public void setPlanName(String planName) { this.planName = planName; }
    public void setType(CashPlanType type) { this.type = type; }
    public void setPeriod(CashPlanPeriod period) { this.period = period; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    public void setStatus(CashPlanStatus status) { this.status = status; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public void setInflowAmount(BigDecimal inflowAmount) { this.inflowAmount = inflowAmount; }
    public void setOutflowAmount(BigDecimal outflowAmount) { this.outflowAmount = outflowAmount; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public void setApprovedBy(Long approvedBy) { this.approvedBy = approvedBy; }
    public void setApproverName(String approverName) { this.approverName = approverName; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setLines(List<CashPlanLine> lines) { this.lines = lines; }
}