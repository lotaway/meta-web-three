package com.metawebthree.reporting.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class InventoryReport {
    private Long id;
    private String reportNo;
    private ReportType type;
    private LocalDateTime reportDate;
    private BigDecimal totalInventoryValue;
    private Integer totalSkuCount;
    private Integer totalQuantity;
    private BigDecimal turnoverRate;
    private BigDecimal slowMovingRate;
    private Integer slowMovingCount;
    private String warehouseBreakdown;
    private String categoryBreakdown;
    private String lowStockItems;
    private LocalDateTime createdAt;

    public enum ReportType {
        DAILY, WEEKLY, MONTHLY
    }

    public void generateDailyReport(LocalDateTime reportDate) {
        this.reportNo = "INV-" + reportDate.toLocalDate().toString().replace("-", "");
        this.type = ReportType.DAILY;
        this.reportDate = reportDate;
        this.createdAt = LocalDateTime.now();
    }

    public void generateMonthlyReport(int year, int month) {
        this.reportNo = "INV-" + year + String.format("%02d", month);
        this.type = ReportType.MONTHLY;
        this.reportDate = LocalDateTime.of(year, month, 28, 0, 0);
        this.createdAt = LocalDateTime.now();
    }

    public void setMetrics(BigDecimal totalInventoryValue, Integer totalSkuCount, Integer totalQuantity,
                          BigDecimal turnoverRate, BigDecimal slowMovingRate, Integer slowMovingCount) {
        this.totalInventoryValue = totalInventoryValue;
        this.totalSkuCount = totalSkuCount;
        this.totalQuantity = totalQuantity;
        this.turnoverRate = turnoverRate;
        this.slowMovingRate = slowMovingRate;
        this.slowMovingCount = slowMovingCount;
    }

    public void setWarehouseBreakdown(String breakdown) {
        this.warehouseBreakdown = breakdown;
    }

    public void setCategoryBreakdown(String breakdown) {
        this.categoryBreakdown = breakdown;
    }

    public void setLowStockItems(String items) {
        this.lowStockItems = items;
    }

    public Long getId() { return id; }
    public String getReportNo() { return reportNo; }
    public ReportType getType() { return type; }
    public LocalDateTime getReportDate() { return reportDate; }
    public BigDecimal getTotalInventoryValue() { return totalInventoryValue; }
    public Integer getTotalSkuCount() { return totalSkuCount; }
    public Integer getTotalQuantity() { return totalQuantity; }
    public BigDecimal getTurnoverRate() { return turnoverRate; }
    public BigDecimal getSlowMovingRate() { return slowMovingRate; }
    public Integer getSlowMovingCount() { return slowMovingCount; }
    public String getWarehouseBreakdown() { return warehouseBreakdown; }
    public String getCategoryBreakdown() { return categoryBreakdown; }
    public String getLowStockItems() { return lowStockItems; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setId(Long id) { this.id = id; }
    public void setTotalInventoryValue(BigDecimal totalInventoryValue) { this.totalInventoryValue = totalInventoryValue; }
    public void setTotalSkuCount(Integer totalSkuCount) { this.totalSkuCount = totalSkuCount; }
    public void setTotalQuantity(Integer totalQuantity) { this.totalQuantity = totalQuantity; }
    public void setTurnoverRate(BigDecimal turnoverRate) { this.turnoverRate = turnoverRate; }
    public void setSlowMovingRate(BigDecimal slowMovingRate) { this.slowMovingRate = slowMovingRate; }
    public void setSlowMovingCount(Integer slowMovingCount) { this.slowMovingCount = slowMovingCount; }
    public void setReportNo(String reportNo) { this.reportNo = reportNo; }
    public void setType(ReportType type) { this.type = type; }
    public void setReportDate(LocalDateTime reportDate) { this.reportDate = reportDate; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}