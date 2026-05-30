package com.metawebthree.finance.application.command.asset.dto;

import java.math.BigDecimal;

public class FixedAssetCreateCommand {
    private Long id;
    private String assetCode;
    private String assetName;
    private String assetCategory;
    private String specification;
    private String model;
    private String serialNumber;
    private Long supplierId;
    private String supplierName;
    private String manufacturer;
    private String purchaseDate;
    private BigDecimal originalValue;
    private BigDecimal residualValue;
    private Integer usefulLife;
    private String depreciationMethod;
    private Long departmentId;
    private String departmentName;
    private String location;
    private String custodian;
    private String remark;
    private Long createdBy;
    private String creatorName;

    public Long getId() { return id; }
    public String getAssetCode() { return assetCode; }
    public String getAssetName() { return assetName; }
    public String getAssetCategory() { return assetCategory; }
    public String getSpecification() { return specification; }
    public String getModel() { return model; }
    public String getSerialNumber() { return serialNumber; }
    public Long getSupplierId() { return supplierId; }
    public String getSupplierName() { return supplierName; }
    public String getManufacturer() { return manufacturer; }
    public String getPurchaseDate() { return purchaseDate; }
    public BigDecimal getOriginalValue() { return originalValue; }
    public BigDecimal getResidualValue() { return residualValue; }
    public Integer getUsefulLife() { return usefulLife; }
    public String getDepreciationMethod() { return depreciationMethod; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public String getLocation() { return location; }
    public String getCustodian() { return custodian; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }

    public void setId(Long id) { this.id = id; }
    public void setAssetCode(String assetCode) { this.assetCode = assetCode; }
    public void setAssetName(String assetName) { this.assetName = assetName; }
    public void setAssetCategory(String assetCategory) { this.assetCategory = assetCategory; }
    public void setSpecification(String specification) { this.specification = specification; }
    public void setModel(String model) { this.model = model; }
    public void setSerialNumber(String serialNumber) { this.serialNumber = serialNumber; }
    public void setSupplierId(Long supplierId) { this.supplierId = supplierId; }
    public void setSupplierName(String supplierName) { this.supplierName = supplierName; }
    public void setManufacturer(String manufacturer) { this.manufacturer = manufacturer; }
    public void setPurchaseDate(String purchaseDate) { this.purchaseDate = purchaseDate; }
    public void setOriginalValue(BigDecimal originalValue) { this.originalValue = originalValue; }
    public void setResidualValue(BigDecimal residualValue) { this.residualValue = residualValue; }
    public void setUsefulLife(Integer usefulLife) { this.usefulLife = usefulLife; }
    public void setDepreciationMethod(String depreciationMethod) { this.depreciationMethod = depreciationMethod; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setLocation(String location) { this.location = location; }
    public void setCustodian(String custodian) { this.custodian = custodian; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
}