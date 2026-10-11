package com.metawebthree.dataanalysis.application.dto;

public class InventoryAnalysisDTO {
    private String productId;
    private String productName;
    private String category;
    private Long currentStock;
    private Long safetyStock;
    private Long reorderPoint;
    private Integer turnoverRate;
    private Integer stockDays;
    private String stockStatus;

    public String getProductId() {
        return productId;
    }

    public void setProductId(String productId) {
        this.productId = productId;
    }

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public Long getCurrentStock() {
        return currentStock;
    }

    public void setCurrentStock(Long currentStock) {
        this.currentStock = currentStock;
    }

    public Long getSafetyStock() {
        return safetyStock;
    }

    public void setSafetyStock(Long safetyStock) {
        this.safetyStock = safetyStock;
    }

    public Long getReorderPoint() {
        return reorderPoint;
    }

    public void setReorderPoint(Long reorderPoint) {
        this.reorderPoint = reorderPoint;
    }

    public Integer getTurnoverRate() {
        return turnoverRate;
    }

    public void setTurnoverRate(Integer turnoverRate) {
        this.turnoverRate = turnoverRate;
    }

    public Integer getStockDays() {
        return stockDays;
    }

    public void setStockDays(Integer stockDays) {
        this.stockDays = stockDays;
    }

    public String getStockStatus() {
        return stockStatus;
    }

    public void setStockStatus(String stockStatus) {
        this.stockStatus = stockStatus;
    }
}