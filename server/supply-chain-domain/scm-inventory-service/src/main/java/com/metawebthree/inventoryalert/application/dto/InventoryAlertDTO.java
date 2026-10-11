package com.metawebthree.inventoryalert.application.dto;

public class InventoryAlertDTO {
    private Long id;
    private Long productId;
    private Long skuId;
    private String productName;
    private String skuCode;
    private Long warehouseId;
    private String warehouseName;
    private Integer currentStock;
    private Integer threshold;
    private Integer alertLevel;
    private String alertLevelDesc;
    private Integer alertStatus;
    private String alertStatusDesc;
    private String alertMessage;
    private String alertTime;
    private String resolvedTime;
    private String remark;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getProductId() {
        return productId;
    }

    public void setProductId(Long productId) {
        this.productId = productId;
    }

    public Long getSkuId() {
        return skuId;
    }

    public void setSkuId(Long skuId) {
        this.skuId = skuId;
    }

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public String getSkuCode() {
        return skuCode;
    }

    public void setSkuCode(String skuCode) {
        this.skuCode = skuCode;
    }

    public Long getWarehouseId() {
        return warehouseId;
    }

    public void setWarehouseId(Long warehouseId) {
        this.warehouseId = warehouseId;
    }

    public String getWarehouseName() {
        return warehouseName;
    }

    public void setWarehouseName(String warehouseName) {
        this.warehouseName = warehouseName;
    }

    public Integer getCurrentStock() {
        return currentStock;
    }

    public void setCurrentStock(Integer currentStock) {
        this.currentStock = currentStock;
    }

    public Integer getThreshold() {
        return threshold;
    }

    public void setThreshold(Integer threshold) {
        this.threshold = threshold;
    }

    public Integer getAlertLevel() {
        return alertLevel;
    }

    public void setAlertLevel(Integer alertLevel) {
        this.alertLevel = alertLevel;
    }

    public String getAlertLevelDesc() {
        return alertLevelDesc;
    }

    public void setAlertLevelDesc(String alertLevelDesc) {
        this.alertLevelDesc = alertLevelDesc;
    }

    public Integer getAlertStatus() {
        return alertStatus;
    }

    public void setAlertStatus(Integer alertStatus) {
        this.alertStatus = alertStatus;
    }

    public String getAlertStatusDesc() {
        return alertStatusDesc;
    }

    public void setAlertStatusDesc(String alertStatusDesc) {
        this.alertStatusDesc = alertStatusDesc;
    }

    public String getAlertMessage() {
        return alertMessage;
    }

    public void setAlertMessage(String alertMessage) {
        this.alertMessage = alertMessage;
    }

    public String getAlertTime() {
        return alertTime;
    }

    public void setAlertTime(String alertTime) {
        this.alertTime = alertTime;
    }

    public String getResolvedTime() {
        return resolvedTime;
    }

    public void setResolvedTime(String resolvedTime) {
        this.resolvedTime = resolvedTime;
    }

    public String getRemark() {
        return remark;
    }

    public void setRemark(String remark) {
        this.remark = remark;
    }
}