package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class ResourcePool {
    private Long id;
    private String poolCode;
    private String poolName;
    private Long costCenterId;
    private String costCenterName;
    private ResourcePoolType type;
    private BigDecimal totalBudget;
    private BigDecimal allocatedAmount;
    private BigDecimal usedAmount;
    private String currency;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ResourcePoolType {
        LABOR, MATERIAL, MACHINE, OVERHEAD, ENERGY, MAINTENANCE, QUALITY, STORAGE
    }

    public void create(String poolCode, String poolName, Long costCenterId,
                       String costCenterName, ResourcePoolType type,
                       BigDecimal totalBudget, String currency, String description) {
        this.poolCode = poolCode;
        this.poolName = poolName;
        this.costCenterId = costCenterId;
        this.costCenterName = costCenterName;
        this.type = type;
        this.totalBudget = totalBudget;
        this.allocatedAmount = BigDecimal.ZERO;
        this.usedAmount = BigDecimal.ZERO;
        this.currency = currency;
        this.description = description;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void allocate(BigDecimal amount) {
        this.allocatedAmount = this.allocatedAmount.add(amount);
        this.updatedAt = LocalDateTime.now();
    }

    public void use(BigDecimal amount) {
        this.usedAmount = this.usedAmount.add(amount);
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal getAvailableAmount() {
        return totalBudget.subtract(allocatedAmount);
    }

    public BigDecimal getUtilizationRate() {
        if (totalBudget == null || totalBudget.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return allocatedAmount.divide(totalBudget, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(new BigDecimal("100"));
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPoolCode() { return poolCode; }
    public void setPoolCode(String poolCode) { this.poolCode = poolCode; }
    public String getPoolName() { return poolName; }
    public void setPoolName(String poolName) { this.poolName = poolName; }
    public Long getCostCenterId() { return costCenterId; }
    public void setCostCenterId(Long costCenterId) { this.costCenterId = costCenterId; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public ResourcePoolType getType() { return type; }
    public void setType(ResourcePoolType type) { this.type = type; }
    public BigDecimal getTotalBudget() { return totalBudget; }
    public void setTotalBudget(BigDecimal totalBudget) { this.totalBudget = totalBudget; }
    public BigDecimal getAllocatedAmount() { return allocatedAmount; }
    public void setAllocatedAmount(BigDecimal allocatedAmount) { this.allocatedAmount = allocatedAmount; }
    public BigDecimal getUsedAmount() { return usedAmount; }
    public void setUsedAmount(BigDecimal usedAmount) { this.usedAmount = usedAmount; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}