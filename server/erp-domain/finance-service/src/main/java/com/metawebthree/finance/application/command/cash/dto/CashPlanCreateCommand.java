package com.metawebthree.finance.application.command.cash.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

public class CashPlanCreateCommand {
    private String planCode;
    private String planName;
    private String type;
    private String period;
    private LocalDate startDate;
    private LocalDate endDate;
    private Long departmentId;
    private String departmentName;
    private Long createdBy;
    private String creatorName;
    private String remark;
    private List<CashPlanLineCreateCommand> lines;

    // Getters and Setters
    public String getPlanCode() { return planCode; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public String getPlanName() { return planName; }
    public void setPlanName(String planName) { this.planName = planName; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getPeriod() { return period; }
    public void setPeriod(String period) { this.period = period; }
    public LocalDate getStartDate() { return startDate; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    public LocalDate getEndDate() { return endDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    public Long getDepartmentId() { return departmentId; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public String getDepartmentName() { return departmentName; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public List<CashPlanLineCreateCommand> getLines() { return lines; }
    public void setLines(List<CashPlanLineCreateCommand> lines) { this.lines = lines; }
}

class CashPlanLineCreateCommand {
    private String categoryCode;
    private String categoryName;
    private String flowDirection;
    private BigDecimal plannedAmount;
    private LocalDate plannedDate;
    private String remark;

    public String getCategoryCode() { return categoryCode; }
    public void setCategoryCode(String categoryCode) { this.categoryCode = categoryCode; }
    public String getCategoryName() { return categoryName; }
    public void setCategoryName(String categoryName) { this.categoryName = categoryName; }
    public String getFlowDirection() { return flowDirection; }
    public void setFlowDirection(String flowDirection) { this.flowDirection = flowDirection; }
    public BigDecimal getPlannedAmount() { return plannedAmount; }
    public void setPlannedAmount(BigDecimal plannedAmount) { this.plannedAmount = plannedAmount; }
    public LocalDate getPlannedDate() { return plannedDate; }
    public void setPlannedDate(LocalDate plannedDate) { this.plannedDate = plannedDate; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}