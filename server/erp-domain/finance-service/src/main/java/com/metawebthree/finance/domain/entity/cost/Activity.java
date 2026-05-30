package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class Activity {
    private Long id;
    private String activityCode;
    private String activityName;
    private Long costCenterId;
    private String costCenterName;
    private Long resourcePoolId;
    private String resourcePoolName;
    private Long costDriverId;
    private String costDriverCode;
    private String costDriverName;
    private ActivityType type;
    private BigDecimal totalCost;
    private BigDecimal driverQuantity;
    private BigDecimal driverRate;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ActivityType {
        UNIT_LEVEL, BATCH_LEVEL, PRODUCT_LEVEL, FACILITY_LEVEL
    }

    public void create(String activityCode, String activityName, Long costCenterId,
                       String costCenterName, Long resourcePoolId, String resourcePoolName,
                       Long costDriverId, String costDriverCode, String costDriverName,
                       ActivityType type, String description) {
        this.activityCode = activityCode;
        this.activityName = activityName;
        this.costCenterId = costCenterId;
        this.costCenterName = costCenterName;
        this.resourcePoolId = resourcePoolId;
        this.resourcePoolName = resourcePoolName;
        this.costDriverId = costDriverId;
        this.costDriverCode = costDriverCode;
        this.costDriverName = costDriverName;
        this.type = type;
        this.totalCost = BigDecimal.ZERO;
        this.driverQuantity = BigDecimal.ZERO;
        this.description = description;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void assignCost(BigDecimal cost, BigDecimal driverQuantity) {
        this.totalCost = this.totalCost.add(cost);
        this.driverQuantity = this.driverQuantity.add(driverQuantity);
        if (this.driverQuantity.compareTo(BigDecimal.ZERO) > 0) {
            this.driverRate = this.totalCost.divide(this.driverQuantity, 4, BigDecimal.ROUND_HALF_UP);
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void calculateRate() {
        if (driverQuantity != null && driverQuantity.compareTo(BigDecimal.ZERO) > 0) {
            this.driverRate = totalCost.divide(driverQuantity, 4, BigDecimal.ROUND_HALF_UP);
        }
    }

    public BigDecimal getRate() {
        return driverRate != null ? driverRate : BigDecimal.ZERO;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getActivityCode() { return activityCode; }
    public void setActivityCode(String activityCode) { this.activityCode = activityCode; }
    public String getActivityName() { return activityName; }
    public void setActivityName(String activityName) { this.activityName = activityName; }
    public Long getCostCenterId() { return costCenterId; }
    public void setCostCenterId(Long costCenterId) { this.costCenterId = costCenterId; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public Long getResourcePoolId() { return resourcePoolId; }
    public void setResourcePoolId(Long resourcePoolId) { this.resourcePoolId = resourcePoolId; }
    public String getResourcePoolName() { return resourcePoolName; }
    public void setResourcePoolName(String resourcePoolName) { this.resourcePoolName = resourcePoolName; }
    public Long getCostDriverId() { return costDriverId; }
    public void setCostDriverId(Long costDriverId) { this.costDriverId = costDriverId; }
    public String getCostDriverCode() { return costDriverCode; }
    public void setCostDriverCode(String costDriverCode) { this.costDriverCode = costDriverCode; }
    public String getCostDriverName() { return costDriverName; }
    public void setCostDriverName(String costDriverName) { this.costDriverName = costDriverName; }
    public ActivityType getType() { return type; }
    public void setType(ActivityType type) { this.type = type; }
    public BigDecimal getTotalCost() { return totalCost; }
    public void setTotalCost(BigDecimal totalCost) { this.totalCost = totalCost; }
    public BigDecimal getDriverQuantity() { return driverQuantity; }
    public void setDriverQuantity(BigDecimal driverQuantity) { this.driverQuantity = driverQuantity; }
    public BigDecimal getDriverRate() { return driverRate; }
    public void setDriverRate(BigDecimal driverRate) { this.driverRate = driverRate; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}