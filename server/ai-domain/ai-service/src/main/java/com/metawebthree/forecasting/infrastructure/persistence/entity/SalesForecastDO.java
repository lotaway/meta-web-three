package com.metawebthree.forecasting.infrastructure.persistence.entity;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class SalesForecastDO {
    private Long id;
    private String skuCode;
    private String skuName;
    private Long warehouseId;
    private LocalDate forecastDate;
    private Integer forecastQuantity;
    private Integer actualQuantity;
    private BigDecimal forecastAmount;
    private BigDecimal actualAmount;
    private String status;
    private String forecastModel;
    private BigDecimal confidenceLevel;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getSkuCode() { return skuCode; }
    public void setSkuCode(String skuCode) { this.skuCode = skuCode; }
    public String getSkuName() { return skuName; }
    public void setSkuName(String skuName) { this.skuName = skuName; }
    public Long getWarehouseId() { return warehouseId; }
    public void setWarehouseId(Long warehouseId) { this.warehouseId = warehouseId; }
    public LocalDate getForecastDate() { return forecastDate; }
    public void setForecastDate(LocalDate forecastDate) { this.forecastDate = forecastDate; }
    public Integer getForecastQuantity() { return forecastQuantity; }
    public void setForecastQuantity(Integer forecastQuantity) { this.forecastQuantity = forecastQuantity; }
    public Integer getActualQuantity() { return actualQuantity; }
    public void setActualQuantity(Integer actualQuantity) { this.actualQuantity = actualQuantity; }
    public BigDecimal getForecastAmount() { return forecastAmount; }
    public void setForecastAmount(BigDecimal forecastAmount) { this.forecastAmount = forecastAmount; }
    public BigDecimal getActualAmount() { return actualAmount; }
    public void setActualAmount(BigDecimal actualAmount) { this.actualAmount = actualAmount; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getForecastModel() { return forecastModel; }
    public void setForecastModel(String forecastModel) { this.forecastModel = forecastModel; }
    public BigDecimal getConfidenceLevel() { return confidenceLevel; }
    public void setConfidenceLevel(BigDecimal confidenceLevel) { this.confidenceLevel = confidenceLevel; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}