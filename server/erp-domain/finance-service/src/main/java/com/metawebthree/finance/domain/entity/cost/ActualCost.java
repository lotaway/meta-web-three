package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class ActualCost {
    private Long id;
    private String productCode;
    private String productName;
    private String productionOrderNo;
    private Long costCenterId;
    private String costCenterName;
    private LocalDate costDate;
    private BigDecimal actualMaterialCost;
    private BigDecimal actualLaborCost;
    private BigDecimal actualOverheadCost;
    private BigDecimal actualTotalCost;
    private BigDecimal quantity;
    private String unit;
    private String currency;
    private String costType;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Long createdBy;

    public void create(String productCode, String productName, String productionOrderNo,
                       Long costCenterId, String costCenterName, LocalDate costDate,
                       BigDecimal actualMaterialCost, BigDecimal actualLaborCost,
                       BigDecimal actualOverheadCost, BigDecimal quantity, String unit,
                       String costType, Long createdBy, String currency) {
        this.productCode = productCode;
        this.productName = productName;
        this.productionOrderNo = productionOrderNo;
        this.costCenterId = costCenterId;
        this.costCenterName = costCenterName;
        this.costDate = costDate;
        this.actualMaterialCost = actualMaterialCost;
        this.actualLaborCost = actualLaborCost;
        this.actualOverheadCost = actualOverheadCost;
        this.quantity = quantity;
        this.unit = unit;
        this.costType = costType;
        this.createdBy = createdBy;
        this.currency = currency;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        calculateTotal();
    }

    public void calculateTotal() {
        this.actualTotalCost = actualMaterialCost
                .add(actualLaborCost)
                .add(actualOverheadCost);
    }

    public BigDecimal getUnitCost() {
        if (quantity == null || quantity.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return actualTotalCost.divide(quantity, 4, RoundingMode.HALF_UP);
    }

    public void updateActualCost(BigDecimal materialCost, BigDecimal laborCost, BigDecimal overheadCost) {
        this.actualMaterialCost = materialCost;
        this.actualLaborCost = laborCost;
        this.actualOverheadCost = overheadCost;
        this.updatedAt = LocalDateTime.now();
        calculateTotal();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getProductionOrderNo() { return productionOrderNo; }
    public void setProductionOrderNo(String productionOrderNo) { this.productionOrderNo = productionOrderNo; }
    public Long getCostCenterId() { return costCenterId; }
    public void setCostCenterId(Long costCenterId) { this.costCenterId = costCenterId; }
    public String getCostCenterName() { return costCenterName; }
    public void setCostCenterName(String costCenterName) { this.costCenterName = costCenterName; }
    public LocalDate getCostDate() { return costDate; }
    public void setCostDate(LocalDate costDate) { this.costDate = costDate; }
    public BigDecimal getActualMaterialCost() { return actualMaterialCost; }
    public void setActualMaterialCost(BigDecimal actualMaterialCost) { this.actualMaterialCost = actualMaterialCost; }
    public BigDecimal getActualLaborCost() { return actualLaborCost; }
    public void setActualLaborCost(BigDecimal actualLaborCost) { this.actualLaborCost = actualLaborCost; }
    public BigDecimal getActualOverheadCost() { return actualOverheadCost; }
    public void setActualOverheadCost(BigDecimal actualOverheadCost) { this.actualOverheadCost = actualOverheadCost; }
    public BigDecimal getActualTotalCost() { return actualTotalCost; }
    public void setActualTotalCost(BigDecimal actualTotalCost) { this.actualTotalCost = actualTotalCost; }
    public BigDecimal getQuantity() { return quantity; }
    public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getCostType() { return costType; }
    public void setCostType(String costType) { this.costType = costType; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
}