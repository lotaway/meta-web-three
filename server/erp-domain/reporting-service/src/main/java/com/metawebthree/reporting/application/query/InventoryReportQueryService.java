package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.InventoryReport;
import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.InventoryReportRepository;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class InventoryReportQueryService {

    private static final BigDecimal COST_RATIO = new BigDecimal("0.65");
    private static final BigDecimal INVENTORY_DAYS_COVERAGE = BigDecimal.valueOf(30);
    private static final BigDecimal TURNOVER_ANNUALIZATION = BigDecimal.valueOf(365);
    private static final int DEFAULT_SKU_COUNT = 2500;
    private static final BigDecimal SLOW_MOVING_THRESHOLD = new BigDecimal("0.08");
    private static final BigDecimal MIN_INVENTORY_VALUE_DAILY = BigDecimal.valueOf(10000);
    private static final BigDecimal MIN_INVENTORY_VALUE_MONTHLY = BigDecimal.valueOf(100000);
    private static final BigDecimal AVG_UNIT_PRICE = new BigDecimal("200");
    private static final BigDecimal DAYS_IN_MONTH = BigDecimal.valueOf(30);

    private final InventoryReportRepository repository;
    private final SalesReportRepository salesReportRepository;

    public InventoryReportQueryService(InventoryReportRepository repository, SalesReportRepository salesReportRepository) {
        this.repository = repository;
        this.salesReportRepository = salesReportRepository;
    }

    public void generateDailyReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);

        buildAndSaveDailyReport(now, dayStart, dayEnd);
    }

    public void generateMonthlyReport(int year, int month) {
        LocalDateTime start = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime end = start.plusMonths(1).minusSeconds(1);

        InventoryReport report = InventoryReport.generateMonthlyReport(year, month);
        MonthlyInventoryMetrics metrics = calcMonthlyMetrics(start, end);
        report = report.withMetrics(
                metrics.totalInventoryValue,
                metrics.totalSkuCount,
                metrics.totalQuantity,
                metrics.turnoverRate,
                metrics.slowMovingRate,
                metrics.slowMovingCount
        ).withWarehouseBreakdown(metrics.warehouseBreakdown)
         .withCategoryBreakdown(metrics.categoryBreakdown);

        repository.save(report);
    }

    public Optional<InventoryReport> getById(Long id) {
        return repository.findById(id);
    }

    public List<InventoryReport> listByType(String type) {
        InventoryReport.ReportType t = InventoryReport.ReportType.valueOf(type.toUpperCase());
        return repository.findByType(t);
    }

    public List<InventoryReport> listAll() {
        return repository.findAll();
    }

    private void buildAndSaveDailyReport(LocalDateTime now, LocalDateTime dayStart, LocalDateTime dayEnd) {
        InventoryReport report = InventoryReport.generateDailyReport(now);
        DailyInventoryMetrics metrics = calcDailyMetrics(dayStart, dayEnd);
        report = report.withMetrics(
                metrics.totalInventoryValue,
                metrics.totalSkuCount,
                metrics.totalQuantity,
                metrics.turnoverRate,
                metrics.slowMovingRate,
                metrics.slowMovingCount
        ).withWarehouseBreakdown(metrics.warehouseBreakdown)
         .withCategoryBreakdown(metrics.categoryBreakdown)
         .withLowStockItems(metrics.lowStockItems);

        repository.save(report);
    }

    private BigDecimal totalSalesInPeriod(LocalDateTime start, LocalDateTime end) {
        return salesReportRepository.findByDateRange(start, end).stream()
                .map(SalesReport::getTotalSalesAmount)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private Map<String, BigDecimal> aggregateCategorySales(LocalDateTime start, LocalDateTime end) {
        Map<String, BigDecimal> catSales = new LinkedHashMap<>();
        for (SalesReport sr : salesReportRepository.findByDateRange(start, end)) {
            if (sr.getCategoryBreakdown() != null && !sr.getCategoryBreakdown().isEmpty()) {
                catSales.merge("sales", sr.getTotalSalesAmount() != null ? sr.getTotalSalesAmount() : BigDecimal.ZERO, BigDecimal::add);
            }
        }
        return catSales;
    }

    private String buildCategoryBreakdown(Map<String, BigDecimal> catSales) {
        BigDecimal total = catSales.values().stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        if (total.compareTo(BigDecimal.ZERO) <= 0) return "{}";
        Map<String, String> pct = new LinkedHashMap<>();
        for (Map.Entry<String, BigDecimal> e : catSales.entrySet()) {
            BigDecimal ratio = e.getValue().multiply(BigDecimal.valueOf(100)).divide(total, 1, RoundingMode.HALF_UP);
            pct.put(e.getKey(), ratio.toString() + "%");
        }
        return pct.toString();
    }

    private DailyInventoryMetrics calcDailyMetrics(LocalDateTime start, LocalDateTime end) {
        BigDecimal totalSales = totalSalesInPeriod(start, end);
        BigDecimal cogs = totalSales.multiply(COST_RATIO).setScale(2, RoundingMode.HALF_UP);
        BigDecimal invValue = cogs.multiply(INVENTORY_DAYS_COVERAGE).max(MIN_INVENTORY_VALUE_DAILY);
        int skuCount = estimateSkuCount();
        int totalQty = estimateTotalQuantity(totalSales);
        BigDecimal turnover = calcTurnoverRate(cogs, invValue);
        BigDecimal slowMoving = calcSlowMovingRate(turnover);
        int slowCount = BigDecimal.valueOf(skuCount).multiply(slowMoving).setScale(0, RoundingMode.DOWN).intValue();
        String whBreakdown = buildWarehouseBreakdown();
        String catBreakdown = buildCategoryBreakdown(aggregateCategorySales(start, end));
        String lowStock = buildLowStockItems();
        return new DailyInventoryMetrics(invValue, skuCount, totalQty, turnover, slowMoving, slowCount, whBreakdown, catBreakdown, lowStock);
    }

    private MonthlyInventoryMetrics calcMonthlyMetrics(LocalDateTime start, LocalDateTime end) {
        BigDecimal totalSales = totalSalesInPeriod(start, end);
        BigDecimal dailyAvg = totalSales.divide(DAYS_IN_MONTH, 2, RoundingMode.HALF_UP);
        BigDecimal cogs = dailyAvg.multiply(COST_RATIO).multiply(INVENTORY_DAYS_COVERAGE).setScale(2, RoundingMode.HALF_UP);
        BigDecimal invValue = cogs.max(MIN_INVENTORY_VALUE_MONTHLY);
        int skuCount = estimateSkuCount();
        int totalQty = estimateTotalQuantity(totalSales);
        BigDecimal turnover = calcTurnoverRate(totalSales, invValue);
        BigDecimal slowMoving = calcSlowMovingRate(turnover);
        int slowCount = BigDecimal.valueOf(skuCount).multiply(slowMoving).setScale(0, RoundingMode.DOWN).intValue();
        String whBreakdown = buildWarehouseBreakdown();
        String catBreakdown = buildCategoryBreakdown(aggregateCategorySales(start, end));
        return new MonthlyInventoryMetrics(invValue, skuCount, totalQty, turnover, slowMoving, slowCount, whBreakdown, catBreakdown);
    }

    private int estimateSkuCount() {
        List<InventoryReport> recent = repository.findAll();
        return recent.stream()
                .filter(r -> r.getTotalSkuCount() != null && r.getTotalSkuCount() > 0)
                .mapToInt(InventoryReport::getTotalSkuCount)
                .findFirst().orElse(DEFAULT_SKU_COUNT);
    }

    private int estimateTotalQuantity(BigDecimal totalSales) {
        return totalSales.divide(AVG_UNIT_PRICE, 0, RoundingMode.DOWN).intValue();
    }

    private BigDecimal calcTurnoverRate(BigDecimal cogs, BigDecimal invValue) {
        if (invValue.compareTo(BigDecimal.ZERO) <= 0) return BigDecimal.ZERO;
        return cogs.multiply(TURNOVER_ANNUALIZATION).divide(invValue, 2, RoundingMode.HALF_UP);
    }

    private BigDecimal calcSlowMovingRate(BigDecimal turnover) {
        if (turnover.compareTo(BigDecimal.ONE) <= 0) return BigDecimal.ONE;
        BigDecimal rate = SLOW_MOVING_THRESHOLD.divide(turnover, 4, RoundingMode.HALF_UP);
        return rate.min(BigDecimal.ONE).max(BigDecimal.ZERO);
    }

    private String buildWarehouseBreakdown() {
        return "{}";
    }

    private String buildLowStockItems() {
        return "[]";
    }

    private record DailyInventoryMetrics(
            BigDecimal totalInventoryValue, Integer totalSkuCount, Integer totalQuantity,
            BigDecimal turnoverRate, BigDecimal slowMovingRate, Integer slowMovingCount,
            String warehouseBreakdown, String categoryBreakdown, String lowStockItems) {}

    private record MonthlyInventoryMetrics(
            BigDecimal totalInventoryValue, Integer totalSkuCount, Integer totalQuantity,
            BigDecimal turnoverRate, BigDecimal slowMovingRate, Integer slowMovingCount,
            String warehouseBreakdown, String categoryBreakdown) {}
}
