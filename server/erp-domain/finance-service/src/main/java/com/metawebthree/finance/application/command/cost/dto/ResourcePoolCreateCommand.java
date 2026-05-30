package com.metawebthree.finance.application.command.cost.dto;

import java.math.BigDecimal;

public class ResourcePoolCreateCommand {
    private String poolCode;
    private String poolName;
    private Long costCenterId;
    private String costCenterName;
    private String type;
    private BigDecimal totalBudget;
    private String currency;
    private String description;

    public String getPoolCode() { return poolCode; }
    public void setPoolCode(String poolCode) { this.poolCode = poolCode; }
    public String getPoolName() { return poolName; }
    public void setPoolName(String poolName) { this.poolName = poolName; }
    public Long getCostCenterId() { return costCenterId; }
    public void setCostCenterId(Long costCenterId) { this.costCenterId = costCenterId; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public BigDecimal getTotalBudget() { return totalBudget; }
    public void setTotalBudget(BigDecimal totalBudget) { this.totalBudget = totalBudget; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}