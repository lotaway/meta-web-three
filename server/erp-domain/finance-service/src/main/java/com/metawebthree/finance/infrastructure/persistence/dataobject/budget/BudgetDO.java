package com.metawebthree.finance.infrastructure.persistence.dataobject.budget;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class BudgetDO {
    private Long id;
    private String budgetCode;
    private String budgetName;
    private String type;
    private String period;
    private Long departmentId;
    private String departmentName;
    private String status;
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

    public Long getId() { return id; }
    public String getBudgetCode() { return budgetCode; }
    public String getBudgetName() { return budgetName; }
    public String getType() { return type; }
    public String getPeriod() { return period; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public String getStatus() { return status; }
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

    public void setId(Long id) { this.id = id; }
    public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
    public void setBudgetName(String budgetName) { this.budgetName = budgetName; }
    public void setType(String type) { this.type = type; }
    public void setPeriod(String period) { this.period = period; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setStatus(String status) { this.status = status; }
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
}