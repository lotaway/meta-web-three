package com.metawebthree.finance.application.command.cost.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class ActualCostCreateCommand {
    private String productCode;
    private String productName;
    private String productionOrderNo;
    private Long costCenterId;
    private String costCenterName;
    private LocalDate costDate;
    private BigDecimal actualMaterialCost;
    private BigDecimal actualLaborCost;
    private BigDecimal actualOverheadCost;
    private BigDecimal quantity;
    private String unit;
    private String costType;
    private Long createdBy;
    private String currency;
    private String remark;

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
    public BigDecimal getQuantity() { return quantity; }
    public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public String getCostType() { return costType; }
    public void setCostType(String costType) { this.costType = costType; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}