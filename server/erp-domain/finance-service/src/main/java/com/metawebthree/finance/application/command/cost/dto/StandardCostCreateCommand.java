package com.metawebthree.finance.application.command.cost.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

public class StandardCostCreateCommand {
    private String productCode;
    private String productName;
    private String productCategory;
    private BigDecimal standardMaterialCost;
    private BigDecimal standardLaborCost;
    private BigDecimal standardOverheadCost;
    private BigDecimal standardQuantity;
    private String unit;
    private LocalDate effectiveDate;
    private String version;
    private Long createdBy;
    private String currency;
    private String remark;

    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getProductCategory() { return productCategory; }
    public void setProductCategory(String productCategory) { this.productCategory = productCategory; }
    public BigDecimal getStandardMaterialCost() { return standardMaterialCost; }
    public void setStandardMaterialCost(BigDecimal standardMaterialCost) { this.standardMaterialCost = standardMaterialCost; }
    public BigDecimal getStandardLaborCost() { return standardLaborCost; }
    public void setStandardLaborCost(BigDecimal standardLaborCost) { this.standardLaborCost = standardLaborCost; }
    public BigDecimal getStandardOverheadCost() { return standardOverheadCost; }
    public void setStandardOverheadCost(BigDecimal standardOverheadCost) { this.standardOverheadCost = standardOverheadCost; }
    public BigDecimal getStandardQuantity() { return standardQuantity; }
    public void setStandardQuantity(BigDecimal standardQuantity) { this.standardQuantity = standardQuantity; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public LocalDate getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDate effectiveDate) { this.effectiveDate = effectiveDate; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}