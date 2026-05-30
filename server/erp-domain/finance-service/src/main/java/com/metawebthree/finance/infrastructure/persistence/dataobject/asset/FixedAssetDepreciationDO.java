package com.metawebthree.finance.infrastructure.persistence.dataobject.asset;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class FixedAssetDepreciationDO {
    private Long id;
    private Long assetId;
    private String assetCode;
    private String assetName;
    private String depreciationPeriod;
    private String depreciationMethod;
    private BigDecimal originalValue;
    private BigDecimal residualValue;
    private Integer usefulLife;
    private BigDecimal depreciationAmount;
    private BigDecimal accumulatedDepreciation;
    private BigDecimal netBookValue;
    private String depreciationDate;
    private String status;
    private String voucherNumber;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public Long getAssetId() { return assetId; }
    public String getAssetCode() { return assetCode; }
    public String getAssetName() { return assetName; }
    public String getDepreciationPeriod() { return depreciationPeriod; }
    public String getDepreciationMethod() { return depreciationMethod; }
    public BigDecimal getOriginalValue() { return originalValue; }
    public BigDecimal getResidualValue() { return residualValue; }
    public Integer getUsefulLife() { return usefulLife; }
    public BigDecimal getDepreciationAmount() { return depreciationAmount; }
    public BigDecimal getAccumulatedDepreciation() { return accumulatedDepreciation; }
    public BigDecimal getNetBookValue() { return netBookValue; }
    public String getDepreciationDate() { return depreciationDate; }
    public String getStatus() { return status; }
    public String getVoucherNumber() { return voucherNumber; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setAssetId(Long assetId) { this.assetId = assetId; }
    public void setAssetCode(String assetCode) { this.assetCode = assetCode; }
    public void setAssetName(String assetName) { this.assetName = assetName; }
    public void setDepreciationPeriod(String depreciationPeriod) { this.depreciationPeriod = depreciationPeriod; }
    public void setDepreciationMethod(String depreciationMethod) { this.depreciationMethod = depreciationMethod; }
    public void setOriginalValue(BigDecimal originalValue) { this.originalValue = originalValue; }
    public void setResidualValue(BigDecimal residualValue) { this.residualValue = residualValue; }
    public void setUsefulLife(Integer usefulLife) { this.usefulLife = usefulLife; }
    public void setDepreciationAmount(BigDecimal depreciationAmount) { this.depreciationAmount = depreciationAmount; }
    public void setAccumulatedDepreciation(BigDecimal accumulatedDepreciation) { this.accumulatedDepreciation = accumulatedDepreciation; }
    public void setNetBookValue(BigDecimal netBookValue) { this.netBookValue = netBookValue; }
    public void setDepreciationDate(String depreciationDate) { this.depreciationDate = depreciationDate; }
    public void setStatus(String status) { this.status = status; }
    public void setVoucherNumber(String voucherNumber) { this.voucherNumber = voucherNumber; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}