package com.metawebthree.forecasting.domain.entity;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "tb_sales_forecast")
public class SalesForecast {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "sku_code", length = 64)
    private String skuCode;

    @Column(name = "sku_name", length = 128)
    private String skuName;

    @Column(name = "warehouse_id")
    private Long warehouseId;

    @Column(name = "forecast_date")
    private LocalDate forecastDate;

    @Column(name = "forecast_quantity")
    private Integer forecastQuantity;

    @Column(name = "actual_quantity")
    private Integer actualQuantity;

    @Column(name = "forecast_amount", precision = 18, scale = 2)
    private BigDecimal forecastAmount;

    @Column(name = "actual_amount", precision = 18, scale = 2)
    private BigDecimal actualAmount;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", length = 32)
    private ForecastStatus status;

    @Column(name = "forecast_model", length = 64)
    private String forecastModel;

    @Column(name = "confidence_level", precision = 5, scale = 2)
    private BigDecimal confidenceLevel;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public enum ForecastStatus {
        PENDING, GENERATED, CONFIRMED, ADJUSTED, ARCHIVED
    }

    public void create(String skuCode, String skuName, Long warehouseId,
                      LocalDate forecastDate, Integer forecastQuantity,
                      String forecastModel, BigDecimal confidenceLevel) {
        this.skuCode = skuCode;
        this.skuName = skuName;
        this.warehouseId = warehouseId;
        this.forecastDate = forecastDate;
        this.forecastQuantity = forecastQuantity;
        this.forecastModel = forecastModel;
        this.confidenceLevel = confidenceLevel;
        this.status = ForecastStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void confirm() {
        if (status != ForecastStatus.PENDING && status != ForecastStatus.GENERATED) {
            throw new IllegalStateException("Cannot confirm forecast in status: " + status);
        }
        this.status = ForecastStatus.CONFIRMED;
        this.updatedAt = LocalDateTime.now();
    }

    public void adjust(Integer newQuantity, BigDecimal newAmount) {
        if (status == ForecastStatus.ARCHIVED) {
            throw new IllegalStateException("Cannot adjust archived forecast");
        }
        this.forecastQuantity = newQuantity;
        this.forecastAmount = newAmount;
        this.status = ForecastStatus.ADJUSTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void archive() {
        if (status != ForecastStatus.CONFIRMED) {
            throw new IllegalStateException("Only confirmed forecasts can be archived");
        }
        this.status = ForecastStatus.ARCHIVED;
        this.updatedAt = LocalDateTime.now();
    }

    public void recordActual(Integer actualQuantity, BigDecimal actualAmount) {
        this.actualQuantity = actualQuantity;
        this.actualAmount = actualAmount;
        this.status = ForecastStatus.GENERATED;
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal calculateAccuracy() {
        if (actualQuantity == null || actualQuantity == 0 || forecastQuantity == 0) {
            return BigDecimal.ZERO;
        }
        int diff = Math.abs(actualQuantity - forecastQuantity);
        return BigDecimal.valueOf(100).subtract(
            BigDecimal.valueOf(diff).multiply(BigDecimal.valueOf(100))
                .divide(BigDecimal.valueOf(forecastQuantity), 2, BigDecimal.ROUND_HALF_UP)
        );
    }

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
    public ForecastStatus getStatus() { return status; }
    public void setStatus(ForecastStatus status) { this.status = status; }
    public String getForecastModel() { return forecastModel; }
    public void setForecastModel(String forecastModel) { this.forecastModel = forecastModel; }
    public BigDecimal getConfidenceLevel() { return confidenceLevel; }
    public void setConfidenceLevel(BigDecimal confidenceLevel) { this.confidenceLevel = confidenceLevel; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}
