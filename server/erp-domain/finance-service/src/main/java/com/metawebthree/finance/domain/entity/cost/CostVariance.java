package com.metawebthree.finance.domain.entity.cost;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class CostVariance {
    private Long id;
    private String productCode;
    private String productName;
    private String productionOrderNo;
    private LocalDate varianceDate;
    private BigDecimal standardMaterialCost;
    private BigDecimal actualMaterialCost;
    private BigDecimal materialVariance;
    private BigDecimal materialVarianceRate;
    private BigDecimal standardLaborCost;
    private BigDecimal actualLaborCost;
    private BigDecimal laborVariance;
    private BigDecimal laborVarianceRate;
    private BigDecimal standardOverheadCost;
    private BigDecimal actualOverheadCost;
    private BigDecimal overheadVariance;
    private BigDecimal overheadVarianceRate;
    private BigDecimal standardTotalCost;
    private BigDecimal actualTotalCost;
    private BigDecimal totalVariance;
    private BigDecimal totalVarianceRate;
    private String varianceType;
    private String analysis;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum VarianceType {
        FAVORABLE, UNFAVORABLE, NONE
    }

    public void calculate(String productCode, String productName, String productionOrderNo,
                          LocalDate varianceDate, BigDecimal standardMaterialCost,
                          BigDecimal actualMaterialCost, BigDecimal standardLaborCost,
                          BigDecimal actualLaborCost, BigDecimal standardOverheadCost,
                          BigDecimal actualOverheadCost) {
        this.productCode = productCode;
        this.productName = productName;
        this.productionOrderNo = productionOrderNo;
        this.varianceDate = varianceDate;
        this.standardMaterialCost = standardMaterialCost;
        this.actualMaterialCost = actualMaterialCost;
        this.standardLaborCost = standardLaborCost;
        this.actualLaborCost = actualLaborCost;
        this.standardOverheadCost = standardOverheadCost;
        this.actualOverheadCost = actualOverheadCost;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();

        calculateVariances();
    }

    private void calculateVariances() {
        materialVariance = standardMaterialCost.subtract(actualMaterialCost);
        materialVarianceRate = calculateRate(materialVariance, standardMaterialCost);

        laborVariance = standardLaborCost.subtract(actualLaborCost);
        laborVarianceRate = calculateRate(laborVariance, standardLaborCost);

        overheadVariance = standardOverheadCost.subtract(actualOverheadCost);
        overheadVarianceRate = calculateRate(overheadVariance, standardOverheadCost);

        standardTotalCost = standardMaterialCost.add(standardLaborCost).add(standardOverheadCost);
        actualTotalCost = actualMaterialCost.add(actualLaborCost).add(actualOverheadCost);
        totalVariance = standardTotalCost.subtract(actualTotalCost);
        totalVarianceRate = calculateRate(totalVariance, standardTotalCost);

        if (totalVariance.compareTo(BigDecimal.ZERO) > 0) {
            varianceType = VarianceType.FAVORABLE.name();
        } else if (totalVariance.compareTo(BigDecimal.ZERO) < 0) {
            varianceType = VarianceType.UNFAVORABLE.name();
        } else {
            varianceType = VarianceType.NONE.name();
        }
    }

    private BigDecimal calculateRate(BigDecimal variance, BigDecimal standard) {
        if (standard == null || standard.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return variance.divide(standard, 4, RoundingMode.HALF_UP)
                .multiply(new BigDecimal("100"));
    }

    public void analyze() {
        StringBuilder sb = new StringBuilder();
        if (materialVariance.compareTo(BigDecimal.ZERO) < 0) {
            sb.append("材料成本超支 ").append(materialVariance.abs()).append("; ");
        } else if (materialVariance.compareTo(BigDecimal.ZERO) > 0) {
            sb.append("材料成本节约 ").append(materialVariance).append("; ");
        }

        if (laborVariance.compareTo(BigDecimal.ZERO) < 0) {
            sb.append("人工成本超支 ").append(laborVariance.abs()).append("; ");
        } else if (laborVariance.compareTo(BigDecimal.ZERO) > 0) {
            sb.append("人工成本节约 ").append(laborVariance).append("; ");
        }

        if (overheadVariance.compareTo(BigDecimal.ZERO) < 0) {
            sb.append("制造费用超支 ").append(overheadVariance.abs()).append("; ");
        } else if (overheadVariance.compareTo(BigDecimal.ZERO) > 0) {
            sb.append("制造费用节约 ").append(overheadVariance).append("; ");
        }

        this.analysis = sb.toString();
    }

    public boolean isFavorable() {
        return VarianceType.FAVORABLE.name().equals(varianceType);
    }

    public boolean isUnfavorable() {
        return VarianceType.UNFAVORABLE.name().equals(varianceType);
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getProductionOrderNo() { return productionOrderNo; }
    public void setProductionOrderNo(String productionOrderNo) { this.productionOrderNo = productionOrderNo; }
    public LocalDate getVarianceDate() { return varianceDate; }
    public void setVarianceDate(LocalDate varianceDate) { this.varianceDate = varianceDate; }
    public BigDecimal getStandardMaterialCost() { return standardMaterialCost; }
    public void setStandardMaterialCost(BigDecimal standardMaterialCost) { this.standardMaterialCost = standardMaterialCost; }
    public BigDecimal getActualMaterialCost() { return actualMaterialCost; }
    public void setActualMaterialCost(BigDecimal actualMaterialCost) { this.actualMaterialCost = actualMaterialCost; }
    public BigDecimal getMaterialVariance() { return materialVariance; }
    public void setMaterialVariance(BigDecimal materialVariance) { this.materialVariance = materialVariance; }
    public BigDecimal getMaterialVarianceRate() { return materialVarianceRate; }
    public void setMaterialVarianceRate(BigDecimal materialVarianceRate) { this.materialVarianceRate = materialVarianceRate; }
    public BigDecimal getStandardLaborCost() { return standardLaborCost; }
    public void setStandardLaborCost(BigDecimal standardLaborCost) { this.standardLaborCost = standardLaborCost; }
    public BigDecimal getActualLaborCost() { return actualLaborCost; }
    public void setActualLaborCost(BigDecimal actualLaborCost) { this.actualLaborCost = actualLaborCost; }
    public BigDecimal getLaborVariance() { return laborVariance; }
    public void setLaborVariance(BigDecimal laborVariance) { this.laborVariance = laborVariance; }
    public BigDecimal getLaborVarianceRate() { return laborVarianceRate; }
    public void setLaborVarianceRate(BigDecimal laborVarianceRate) { this.laborVarianceRate = laborVarianceRate; }
    public BigDecimal getStandardOverheadCost() { return standardOverheadCost; }
    public void setStandardOverheadCost(BigDecimal standardOverheadCost) { this.standardOverheadCost = standardOverheadCost; }
    public BigDecimal getActualOverheadCost() { return actualOverheadCost; }
    public void setActualOverheadCost(BigDecimal actualOverheadCost) { this.actualOverheadCost = actualOverheadCost; }
    public BigDecimal getOverheadVariance() { return overheadVariance; }
    public void setOverheadVariance(BigDecimal overheadVariance) { this.overheadVariance = overheadVariance; }
    public BigDecimal getOverheadVarianceRate() { return overheadVarianceRate; }
    public void setOverheadVarianceRate(BigDecimal overheadVarianceRate) { this.overheadVarianceRate = overheadVarianceRate; }
    public BigDecimal getStandardTotalCost() { return standardTotalCost; }
    public void setStandardTotalCost(BigDecimal standardTotalCost) { this.standardTotalCost = standardTotalCost; }
    public BigDecimal getActualTotalCost() { return actualTotalCost; }
    public void setActualTotalCost(BigDecimal actualTotalCost) { this.actualTotalCost = actualTotalCost; }
    public BigDecimal getTotalVariance() { return totalVariance; }
    public void setTotalVariance(BigDecimal totalVariance) { this.totalVariance = totalVariance; }
    public BigDecimal getTotalVarianceRate() { return totalVarianceRate; }
    public void setTotalVarianceRate(BigDecimal totalVarianceRate) { this.totalVarianceRate = totalVarianceRate; }
    public String getVarianceType() { return varianceType; }
    public void setVarianceType(String varianceType) { this.varianceType = varianceType; }
    public String getAnalysis() { return analysis; }
    public void setAnalysis(String analysis) { this.analysis = analysis; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}