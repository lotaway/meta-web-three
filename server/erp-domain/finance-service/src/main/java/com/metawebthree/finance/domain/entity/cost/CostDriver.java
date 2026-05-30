package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class CostDriver {
    private Long id;
    private String driverCode;
    private String driverName;
    private CostDriverType type;
    private String unit;
    private BigDecimal rate;
    private String description;
    private CostDriverStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum CostDriverType {
        laborHours,
        machineHours,
        unitsProduced,
        laborCost,
        materialCost,
        transactions,
        batches,
        orders,
        distance,
        weight
    }

    public enum CostDriverStatus {
        ACTIVE, INACTIVE
    }

    public void create(String driverCode, String driverName, CostDriverType type,
                       String unit, BigDecimal rate, String description) {
        this.driverCode = driverCode;
        this.driverName = driverName;
        this.type = type;
        this.unit = unit;
        this.rate = rate;
        this.description = description;
        this.status = CostDriverStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateRate(BigDecimal newRate) {
        this.rate = newRate;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.status = CostDriverStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDriverCode() { return driverCode; }
    public void setDriverCode(String driverCode) { this.driverCode = driverCode; }
    public String getDriverName() { return driverName; }
    public void setDriverName(String driverName) { this.driverName = driverName; }
    public CostDriverType getType() { return type; }
    public void setType(CostDriverType type) { this.type = type; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public BigDecimal getRate() { return rate; }
    public void setRate(BigDecimal rate) { this.rate = rate; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public CostDriverStatus getStatus() { return status; }
    public void setStatus(CostDriverStatus status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}