package com.metawebthree.finance.application.command.cost.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class CostCenterCreateCommand {
    private String costCenterCode;
    private String costCenterName;
    private String type;
    private Long departmentId;
    private String departmentName;
    private String managerName;
    private BigDecimal budgetAmount;
    private Long createdBy;
    private String description;

    public String getCostCenterCode() { return costCenterCode; }
    public void setCostCenterCode(String costCenterCode) { this.costCenterCode = costCenterCode; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public Long getDepartmentId() { return departmentId; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public String getDepartmentName() { return departmentName; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public String getManagerName() { return managerName; }
    public void setManagerName(String managerName) { this.managerName = managerName; }
    public BigDecimal getBudgetAmount() { return budgetAmount; }
    public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}