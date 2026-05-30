package com.metawebthree.finance.application.command.budget.dto;

import java.math.BigDecimal;
import java.util.List;

public class BudgetCreateCommand {
    private String budgetCode;
    private String budgetName;
    private String type;
    private String period;
    private Long departmentId;
    private String departmentName;
    private Long createdBy;
    private String creatorName;
    private String remark;
    private List<BudgetLineCreateCommand> lines;

    public String getBudgetCode() { return budgetCode; }
    public String getBudgetName() { return budgetName; }
    public String getType() { return type; }
    public String getPeriod() { return period; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public String getRemark() { return remark; }
    public List<BudgetLineCreateCommand> getLines() { return lines; }

    public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
    public void setBudgetName(String budgetName) { this.budgetName = budgetName; }
    public void setType(String type) { this.type = type; }
    public void setPeriod(String period) { this.period = period; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setLines(List<BudgetLineCreateCommand> lines) { this.lines = lines; }
}