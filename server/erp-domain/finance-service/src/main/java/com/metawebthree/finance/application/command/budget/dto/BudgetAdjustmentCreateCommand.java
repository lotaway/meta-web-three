package com.metawebthree.finance.application.command.budget.dto;

import java.math.BigDecimal;

public class BudgetAdjustmentCreateCommand {
    private Long budgetId;
    private String type;
    private String subjectCode;
    private String subjectName;
    private BigDecimal originalAmount;
    private BigDecimal adjustedAmount;
    private Long applicantId;
    private String applicantName;
    private String reason;

    public Long getBudgetId() { return budgetId; }
    public String getType() { return type; }
    public String getSubjectCode() { return subjectCode; }
    public String getSubjectName() { return subjectName; }
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public BigDecimal getAdjustedAmount() { return adjustedAmount; }
    public Long getApplicantId() { return applicantId; }
    public String getApplicantName() { return applicantName; }
    public String getReason() { return reason; }

    public void setBudgetId(Long budgetId) { this.budgetId = budgetId; }
    public void setType(String type) { this.type = type; }
    public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
    public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
    public void setAdjustedAmount(BigDecimal adjustedAmount) { this.adjustedAmount = adjustedAmount; }
    public void setApplicantId(Long applicantId) { this.applicantId = applicantId; }
    public void setApplicantName(String applicantName) { this.applicantName = applicantName; }
    public void setReason(String reason) { this.reason = reason; }
}