package com.metawebthree.reporting.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class InventoryReport {
    private final Long id;
    private final String reportNo;
    private final ReportType type;
    private final LocalDateTime reportDate;
    private final BigDecimal totalInventoryValue;
    private final Integer totalSkuCount;
    private final Integer totalQuantity;
    private final BigDecimal turnoverRate;
    private final BigDecimal slowMovingRate;
    private final Integer slowMovingCount;
    private final String warehouseBreakdown;
    private final String categoryBreakdown;
    private final String lowStockItems;
    private final LocalDateTime createdAt;

    public enum ReportType {
        DAILY, WEEKLY, MONTHLY
    }

    public static InventoryReport generateDailyReport(LocalDateTime reportDate) {
        return new InventoryReport(null, "INV-" + reportDate.toLocalDate().toString().replace("-", ""),
                ReportType.DAILY, reportDate, null, null, null, null, null, null, null, null, null, LocalDateTime.now());
    }

    public static InventoryReport generateMonthlyReport(int year, int month) {
        return new InventoryReport(null, "INV-" + year + String.format("%02d", month),
                ReportType.MONTHLY, LocalDateTime.of(year, month, 28, 0, 0),
                null, null, null, null, null, null, null, null, null, LocalDateTime.now());
    }

    public static InventoryReport createWithAllFields(Long id, String reportNo, ReportType type, LocalDateTime reportDate,
                                                       BigDecimal totalInventoryValue, Integer totalSkuCount, Integer totalQuantity,
                                                       BigDecimal turnoverRate, BigDecimal slowMovingRate, Integer slowMovingCount,
                                                       String warehouseBreakdown, String categoryBreakdown, String lowStockItems,
                                                       LocalDateTime createdAt) {
        return new InventoryReport(id, reportNo, type, reportDate,
                totalInventoryValue, totalSkuCount, totalQuantity,
                turnoverRate, slowMovingRate, slowMovingCount,
                warehouseBreakdown, categoryBreakdown, lowStockItems, createdAt);
    }

    private InventoryReport(Long id, String reportNo, ReportType type, LocalDateTime reportDate,
                            BigDecimal totalInventoryValue, Integer totalSkuCount, Integer totalQuantity,
                            BigDecimal turnoverRate, BigDecimal slowMovingRate, Integer slowMovingCount,
                            String warehouseBreakdown, String categoryBreakdown, String lowStockItems,
                            LocalDateTime createdAt) {
        this.id = id;
        this.reportNo = reportNo;
        this.type = type;
        this.reportDate = reportDate;
        this.totalInventoryValue = totalInventoryValue;
        this.totalSkuCount = totalSkuCount;
        this.totalQuantity = totalQuantity;
        this.turnoverRate = turnoverRate;
        this.slowMovingRate = slowMovingRate;
        this.slowMovingCount = slowMovingCount;
        this.warehouseBreakdown = warehouseBreakdown;
        this.categoryBreakdown = categoryBreakdown;
        this.lowStockItems = lowStockItems;
        this.createdAt = createdAt;
    }

    public InventoryReport withMetrics(
            BigDecimal totalInventoryValue,
            Integer totalSkuCount,
            Integer totalQuantity,
            BigDecimal turnoverRate,
            BigDecimal slowMovingRate,
            Integer slowMovingCount) {
        return new InventoryReport(this.id, this.reportNo, this.type, this.reportDate,
                totalInventoryValue, totalSkuCount, totalQuantity,
                turnoverRate, slowMovingRate, slowMovingCount,
                this.warehouseBreakdown, this.categoryBreakdown, this.lowStockItems, this.createdAt);
    }

    public InventoryReport withWarehouseBreakdown(String breakdown) {
        return new InventoryReport(this.id, this.reportNo, this.type, this.reportDate,
                this.totalInventoryValue, this.totalSkuCount, this.totalQuantity,
                this.turnoverRate, this.slowMovingRate, this.slowMovingCount,
                breakdown, this.categoryBreakdown, this.lowStockItems, this.createdAt);
    }

    public InventoryReport withCategoryBreakdown(String breakdown) {
        return new InventoryReport(this.id, this.reportNo, this.type, this.reportDate,
                this.totalInventoryValue, this.totalSkuCount, this.totalQuantity,
                this.turnoverRate, this.slowMovingRate, this.slowMovingCount,
                this.warehouseBreakdown, breakdown, this.lowStockItems, this.createdAt);
    }

    public InventoryReport withLowStockItems(String items) {
        return new InventoryReport(this.id, this.reportNo, this.type, this.reportDate,
                this.totalInventoryValue, this.totalSkuCount, this.totalQuantity,
                this.turnoverRate, this.slowMovingRate, this.slowMovingCount,
                this.warehouseBreakdown, this.categoryBreakdown, items, this.createdAt);
    }

    public InventoryReport withId(Long id) {
        return new InventoryReport(id, this.reportNo, this.type, this.reportDate,
                this.totalInventoryValue, this.totalSkuCount, this.totalQuantity,
                this.turnoverRate, this.slowMovingRate, this.slowMovingCount,
                this.warehouseBreakdown, this.categoryBreakdown, this.lowStockItems, this.createdAt);
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

    @Override
    public String toString() {
        return "InventoryReport{" +
                "id=" + id +
                ", reportNo='" + reportNo + '\'' +
                ", type=" + type +
                ", reportDate=" + reportDate +
                ", totalInventoryValue=" + totalInventoryValue +
                ", totalSkuCount=" + totalSkuCount +
                ", totalQuantity=" + totalQuantity +
                ", turnoverRate=" + turnoverRate +
                ", slowMovingRate=" + slowMovingRate +
                ", slowMovingCount=" + slowMovingCount +
                ", warehouseBreakdown='" + warehouseBreakdown + '\'' +
                ", categoryBreakdown='" + categoryBreakdown + '\'' +
                ", lowStockItems='" + lowStockItems + '\'' +
                ", createdAt=" + createdAt +
                '}';
    }
}
