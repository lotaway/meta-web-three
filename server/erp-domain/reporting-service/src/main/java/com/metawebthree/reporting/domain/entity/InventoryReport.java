package com.metawebthree.reporting.domain.entity;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Getter
@ToString
@Builder
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
        return InventoryReport.builder()
                .reportNo("INV-" + reportDate.toLocalDate().toString().replace("-", ""))
                .type(ReportType.DAILY)
                .reportDate(reportDate)
                .createdAt(LocalDateTime.now())
                .build();
    }

    public static InventoryReport generateMonthlyReport(int year, int month) {
        return InventoryReport.builder()
                .reportNo("INV-" + year + String.format("%02d", month))
                .type(ReportType.MONTHLY)
                .reportDate(LocalDateTime.of(year, month, 28, 0, 0))
                .createdAt(LocalDateTime.now())
                .build();
    }

    public InventoryReport withMetrics(
            BigDecimal totalInventoryValue,
            Integer totalSkuCount,
            Integer totalQuantity,
            BigDecimal turnoverRate,
            BigDecimal slowMovingRate,
            Integer slowMovingCount) {
        return InventoryReport.builder()
                .id(this.id)
                .reportNo(this.reportNo)
                .type(this.type)
                .reportDate(this.reportDate)
                .totalInventoryValue(totalInventoryValue)
                .totalSkuCount(totalSkuCount)
                .totalQuantity(totalQuantity)
                .turnoverRate(turnoverRate)
                .slowMovingRate(slowMovingRate)
                .slowMovingCount(slowMovingCount)
                .warehouseBreakdown(this.warehouseBreakdown)
                .categoryBreakdown(this.categoryBreakdown)
                .lowStockItems(this.lowStockItems)
                .createdAt(this.createdAt)
                .build();
    }

    public InventoryReport withWarehouseBreakdown(String breakdown) {
        return InventoryReport.builder()
                .id(this.id)
                .reportNo(this.reportNo)
                .type(this.type)
                .reportDate(this.reportDate)
                .totalInventoryValue(this.totalInventoryValue)
                .totalSkuCount(this.totalSkuCount)
                .totalQuantity(this.totalQuantity)
                .turnoverRate(this.turnoverRate)
                .slowMovingRate(this.slowMovingRate)
                .slowMovingCount(this.slowMovingCount)
                .warehouseBreakdown(breakdown)
                .categoryBreakdown(this.categoryBreakdown)
                .lowStockItems(this.lowStockItems)
                .createdAt(this.createdAt)
                .build();
    }

    public InventoryReport withCategoryBreakdown(String breakdown) {
        return InventoryReport.builder()
                .id(this.id)
                .reportNo(this.reportNo)
                .type(this.type)
                .reportDate(this.reportDate)
                .totalInventoryValue(this.totalInventoryValue)
                .totalSkuCount(this.totalSkuCount)
                .totalQuantity(this.totalQuantity)
                .turnoverRate(this.turnoverRate)
                .slowMovingRate(this.slowMovingRate)
                .slowMovingCount(this.slowMovingCount)
                .warehouseBreakdown(this.warehouseBreakdown)
                .categoryBreakdown(breakdown)
                .lowStockItems(this.lowStockItems)
                .createdAt(this.createdAt)
                .build();
    }

    public InventoryReport withLowStockItems(String items) {
        return InventoryReport.builder()
                .id(this.id)
                .reportNo(this.reportNo)
                .type(this.type)
                .reportDate(this.reportDate)
                .totalInventoryValue(this.totalInventoryValue)
                .totalSkuCount(this.totalSkuCount)
                .totalQuantity(this.totalQuantity)
                .turnoverRate(this.turnoverRate)
                .slowMovingRate(this.slowMovingRate)
                .slowMovingCount(this.slowMovingCount)
                .warehouseBreakdown(this.warehouseBreakdown)
                .categoryBreakdown(this.categoryBreakdown)
                .lowStockItems(items)
                .createdAt(this.createdAt)
                .build();
    }

    public InventoryReport withId(Long id) {
        return InventoryReport.builder()
                .id(id)
                .reportNo(this.reportNo)
                .type(this.type)
                .reportDate(this.reportDate)
                .totalInventoryValue(this.totalInventoryValue)
                .totalSkuCount(this.totalSkuCount)
                .totalQuantity(this.totalQuantity)
                .turnoverRate(this.turnoverRate)
                .slowMovingRate(this.slowMovingRate)
                .slowMovingCount(this.slowMovingCount)
                .warehouseBreakdown(this.warehouseBreakdown)
                .categoryBreakdown(this.categoryBreakdown)
                .lowStockItems(this.lowStockItems)
                .createdAt(this.createdAt)
                .build();
    }
}
