package com.metawebthree.finance.application.command.budget.dto;

import java.util.List;

public class BudgetUpdateCommand {
    private Long id;
    private String budgetName;
    private Long departmentId;
    private String departmentName;
    private String remark;
    private List<BudgetLineCreateCommand> lines;

    public Long getId() { return id; }
    public String getBudgetName() { return budgetName; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public String getRemark() { return remark; }
    public List<BudgetLineCreateCommand> getLines() { return lines; }

    public void setId(Long id) { this.id = id; }
    public void setBudgetName(String budgetName) { this.budgetName = budgetName; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setLines(List<BudgetLineCreateCommand> lines) { this.lines = lines; }
}