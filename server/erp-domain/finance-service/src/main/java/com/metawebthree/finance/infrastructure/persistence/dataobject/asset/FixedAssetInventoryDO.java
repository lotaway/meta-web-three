package com.metawebthree.finance.infrastructure.persistence.dataobject.asset;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class FixedAssetInventoryDO {
    private Long id;
    private String inventoryCode;
    private String inventoryName;
    private String inventoryDate;
    private String status;
    private Long departmentId;
    private String departmentName;
    private String inventoryPerson;
    private Long assetId;
    private String assetCode;
    private String assetName;
    private String bookLocation;
    private String actualLocation;
    private String inventoryResult;
    private BigDecimal bookValue;
    private BigDecimal actualValue;
    private String discrepancy;
    private String discrepancyReason;
    private String handleMethod;
    private String handleResult;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public String getInventoryCode() { return inventoryCode; }
    public String getInventoryName() { return inventoryName; }
    public String getInventoryDate() { return inventoryDate; }
    public String getStatus() { return status; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public String getInventoryPerson() { return inventoryPerson; }
    public Long getAssetId() { return assetId; }
    public String getAssetCode() { return assetCode; }
    public String getAssetName() { return assetName; }
    public String getBookLocation() { return bookLocation; }
    public String getActualLocation() { return actualLocation; }
    public String getInventoryResult() { return inventoryResult; }
    public BigDecimal getBookValue() { return bookValue; }
    public BigDecimal getActualValue() { return actualValue; }
    public String getDiscrepancy() { return discrepancy; }
    public String getDiscrepancyReason() { return discrepancyReason; }
    public String getHandleMethod() { return handleMethod; }
    public String getHandleResult() { return handleResult; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setInventoryCode(String inventoryCode) { this.inventoryCode = inventoryCode; }
    public void setInventoryName(String inventoryName) { this.inventoryName = inventoryName; }
    public void setInventoryDate(String inventoryDate) { this.inventoryDate = inventoryDate; }
    public void setStatus(String status) { this.status = status; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setInventoryPerson(String inventoryPerson) { this.inventoryPerson = inventoryPerson; }
    public void setAssetId(Long assetId) { this.assetId = assetId; }
    public void setAssetCode(String assetCode) { this.assetCode = assetCode; }
    public void setAssetName(String assetName) { this.assetName = assetName; }
    public void setBookLocation(String bookLocation) { this.bookLocation = bookLocation; }
    public void setActualLocation(String actualLocation) { this.actualLocation = actualLocation; }
    public void setInventoryResult(String inventoryResult) { this.inventoryResult = inventoryResult; }
    public void setBookValue(BigDecimal bookValue) { this.bookValue = bookValue; }
    public void setActualValue(BigDecimal actualValue) { this.actualValue = actualValue; }
    public void setDiscrepancy(String discrepancy) { this.discrepancy = discrepancy; }
    public void setDiscrepancyReason(String discrepancyReason) { this.discrepancyReason = discrepancyReason; }
    public void setHandleMethod(String handleMethod) { this.handleMethod = handleMethod; }
    public void setHandleResult(String handleResult) { this.handleResult = handleResult; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}