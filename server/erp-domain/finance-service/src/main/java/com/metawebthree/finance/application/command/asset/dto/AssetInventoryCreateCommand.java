package com.metawebthree.finance.application.command.asset.dto;

public class AssetInventoryCreateCommand {
    private String inventoryCode;
    private String inventoryName;
    private String inventoryDate;
    private Long departmentId;
    private String departmentName;
    private String inventoryPerson;
    private Long assetId;
    private String assetCode;
    private String bookLocation;
    private String actualLocation;
    private String inventoryResult;
    private String discrepancyReason;
    private String handleMethod;
    private String remark;
    private Long createdBy;
    private String creatorName;

    public String getInventoryCode() { return inventoryCode; }
    public String getInventoryName() { return inventoryName; }
    public String getInventoryDate() { return inventoryDate; }
    public Long getDepartmentId() { return departmentId; }
    public String getDepartmentName() { return departmentName; }
    public String getInventoryPerson() { return inventoryPerson; }
    public Long getAssetId() { return assetId; }
    public String getAssetCode() { return assetCode; }
    public String getBookLocation() { return bookLocation; }
    public String getActualLocation() { return actualLocation; }
    public String getInventoryResult() { return inventoryResult; }
    public String getDiscrepancyReason() { return discrepancyReason; }
    public String getHandleMethod() { return handleMethod; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }

    public void setInventoryCode(String inventoryCode) { this.inventoryCode = inventoryCode; }
    public void setInventoryName(String inventoryName) { this.inventoryName = inventoryName; }
    public void setInventoryDate(String inventoryDate) { this.inventoryDate = inventoryDate; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
    public void setDepartmentName(String departmentName) { this.departmentName = departmentName; }
    public void setInventoryPerson(String inventoryPerson) { this.inventoryPerson = inventoryPerson; }
    public void setAssetId(Long assetId) { this.assetId = assetId; }
    public void setAssetCode(String assetCode) { this.assetCode = assetCode; }
    public void setBookLocation(String bookLocation) { this.bookLocation = bookLocation; }
    public void setActualLocation(String actualLocation) { this.actualLocation = actualLocation; }
    public void setInventoryResult(String inventoryResult) { this.inventoryResult = inventoryResult; }
    public void setDiscrepancyReason(String discrepancyReason) { this.discrepancyReason = discrepancyReason; }
    public void setHandleMethod(String handleMethod) { this.handleMethod = handleMethod; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
}