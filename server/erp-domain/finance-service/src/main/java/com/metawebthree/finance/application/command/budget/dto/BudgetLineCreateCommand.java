package com.metawebthree.finance.application.command.budget.dto;

import java.math.BigDecimal;

public class BudgetLineCreateCommand {
    private String subjectCode;
    private String subjectName;
    private BigDecimal budgetAmount;
    private String remark;

    public String getSubjectCode() { return subjectCode; }
    public String getSubjectName() { return subjectName; }
    public BigDecimal getBudgetAmount() { return budgetAmount; }
    public String getRemark() { return remark; }

    public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
    public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
    public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
    public void setRemark(String remark) { this.remark = remark; }
}