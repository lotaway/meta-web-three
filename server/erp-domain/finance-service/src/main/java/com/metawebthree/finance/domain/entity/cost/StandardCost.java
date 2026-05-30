package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class StandardCost {
    private Long id;
    private String productCode;
    private String productName;
    private String productCategory;
    private BigDecimal standardMaterialCost;
    private BigDecimal standardLaborCost;
    private BigDecimal standardOverheadCost;
    private BigDecimal standardTotalCost;
    private BigDecimal standardQuantity;
    private String unit;
    private LocalDate effectiveDate;
    private LocalDate expirationDate;
    private StandardCostStatus status;
    private String version;
    private String currency;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Long createdBy;

    public enum StandardCostStatus {
        DRAFT, ACTIVE, OBSOLETE
    }

    public void create(String productCode, String productName, String productCategory,
                       BigDecimal standardMaterialCost, BigDecimal standardLaborCost,
                       BigDecimal standardOverheadCost, BigDecimal standardQuantity,
                       String unit, LocalDate effectiveDate, String version,
                       Long createdBy, String currency) {
        this.productCode = productCode;
        this.productName = productName;
        this.productCategory = productCategory;
        this.standardMaterialCost = standardMaterialCost;
        this.standardLaborCost = standardLaborCost;
        this.standardOverheadCost = standardOverheadCost;
        this.standardQuantity = standardQuantity;
        this.unit = unit;
        this.effectiveDate = effectiveDate;
        this.version = version;
        this.status = StandardCostStatus.DRAFT;
        this.currency = currency;
        this.createdBy = createdBy;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        calculateTotal();
    }

    public void calculateTotal() {
        this.standardTotalCost = standardMaterialCost
                .add(standardLaborCost)
                .add(standardOverheadCost);
    }

    public void activate() {
        this.status = StandardCostStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void obsolete() {
        this.status = StandardCostStatus.OBSOLETE;
        this.expirationDate = LocalDate.now();
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal getUnitCost() {
        if (standardQuantity == null || standardQuantity.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return standardTotalCost.divide(standardQuantity, 4, BigDecimal.ROUND_HALF_UP);
    }

    public boolean isEffective() {
        LocalDate now = LocalDate.now();
        boolean afterEffective = effectiveDate == null || !now.isBefore(effectiveDate);
        boolean beforeExpiration = expirationDate == null || !now.isAfter(expirationDate);
        return afterEffective && beforeExpiration && status == StandardCostStatus.ACTIVE;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
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
    public BigDecimal getStandardTotalCost() { return standardTotalCost; }
    public void setStandardTotalCost(BigDecimal standardTotalCost) { this.standardTotalCost = standardTotalCost; }
    public BigDecimal getStandardQuantity() { return standardQuantity; }
    public void setStandardQuantity(BigDecimal standardQuantity) { this.standardQuantity = standardQuantity; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public LocalDate getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDate effectiveDate) { this.effectiveDate = effectiveDate; }
    public LocalDate getExpirationDate() { return expirationDate; }
    public void setExpirationDate(LocalDate expirationDate) { this.expirationDate = expirationDate; }
    public StandardCostStatus getStatus() { return status; }
    public void setStatus(StandardCostStatus status) { this.status = status; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
}