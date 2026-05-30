package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CostCenter {
    private Long id;
    private String costCenterCode;
    private String costCenterName;
    private CostCenterType type;
    private Long parentId;
    private String parentPath;
    private Long departmentId;
    private String departmentName;
    private String managerName;
    private CostCenterStatus status;
    private BigDecimal budgetAmount;
    private BigDecimal actualCost;
    private String currency;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Long createdBy;

    public enum CostCenterType {
        PRODUCTION, SUPPORT, ADMINISTRATION, SALES, R_AND_D
    }

    public enum CostCenterStatus {
        ACTIVE, INACTIVE, CLOSED
    }

    public void create(String costCenterCode, String costCenterName, CostCenterType type,
                       Long departmentId, String departmentName, String managerName,
                       BigDecimal budgetAmount, Long createdBy) {
        this.costCenterCode = costCenterCode;
        this.costCenterName = costCenterName;
        this.type = type;
        this.departmentId = departmentId;
        this.departmentName = departmentName;
        this.managerName = managerName;
        this.budgetAmount = budgetAmount;
        this.actualCost = BigDecimal.ZERO;
        this.currency = "CNY";
        this.status = CostCenterStatus.ACTIVE;
        this.createdBy = createdBy;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateActualCost(BigDecimal amount) {
        this.actualCost = this.actualCost.add(amount);
        this.updatedAt = LocalDateTime.now();
    }

    public void close() {
        this.status = CostCenterStatus.CLOSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.status = CostCenterStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal getVariance() {
        if (budgetAmount == null || actualCost == null) {
            return BigDecimal.ZERO;
        }
        return budgetAmount.subtract(actualCost);
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getCostCenterCode() { return costCenterCode; }
    public void setCostCenterCode(String costCenterCode) { this.costCenterCode = costCenterCode; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public CostCenterType getType() { return type; }
    public void setType(CostCenterType type) { this.type = type; }
    public Long getParentId() { return parentId; }
    public void setParentId(Long parentId) { this.parentId = parentId; }
    public String getParentPath() { return parentPath; }
    public void setParentPath(String parentPath) { this.parentPath = parentPath; }
    public Long getDepartmentId() { return departmentId; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public String getDepartmentName() { return departmentName; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public String getManagerName() { return managerName; }
    public void setManagerName(String managerName) { this.managerName = managerName; }
    public CostCenterStatus getStatus() { return status; }
    public void setStatus(CostCenterStatus status) { this.status = status; }
    public BigDecimal getBudgetAmount() { return budgetAmount; }
    public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
    public BigDecimal getActualCost() { return actualCost; }
    public void setActualCost(BigDecimal actualCost) { this.actualCost = actualCost; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
}