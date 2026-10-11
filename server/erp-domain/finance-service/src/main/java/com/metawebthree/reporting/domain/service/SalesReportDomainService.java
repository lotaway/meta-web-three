package com.metawebthree.reporting.domain.service;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

public class SalesReportDomainService {

    private static final int PRODUCT_RANKING_LIMIT = 10;

    private final SalesReportRepository repository;
    private final BigDecimal defaultAvgOrderAmount;
    private final BigDecimal grossProfitRateDaily;
    private final BigDecimal grossProfitRateMonthly;

    public SalesReportDomainService(SalesReportRepository repository) {
        this(repository, new BigDecimal("500.00"), new BigDecimal("0.25"), new BigDecimal("0.28"));
    }

    public SalesReportDomainService(SalesReportRepository repository,
                                    BigDecimal defaultAvgOrderAmount,
                                    BigDecimal grossProfitRateDaily,
                                    BigDecimal grossProfitRateMonthly) {
        this.repository = repository;
        this.defaultAvgOrderAmount = defaultAvgOrderAmount;
        this.grossProfitRateDaily = grossProfitRateDaily;
        this.grossProfitRateMonthly = grossProfitRateMonthly;
    }

    public SalesReport generateDailyReport(LocalDateTime now) {
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);

        SalesReport report = new SalesReport();
        report.generateDailyReport(now);

        BigDecimal totalSales = calculateTotalSales(dayStart, dayEnd);
        int orderCount = calculateOrderCountFromHistoricalData(dayStart, dayEnd, totalSales);
        BigDecimal grossProfit = calcGrossProfit(totalSales, grossProfitRateDaily);
        BigDecimal profitMargin = calcProfitMargin(grossProfit, totalSales, orderCount);

        report.setMetrics(totalSales, orderCount, grossProfit, profitMargin);
        report.setCategoryBreakdown(calculateCategoryBreakdown(dayStart, dayEnd));
        report.setChannelBreakdown(calculateChannelBreakdown(dayStart, dayEnd));

        return report;
    }

    public SalesReport generateMonthlyReport(int year, int month) {
        SalesReport report = new SalesReport();
        report.generateMonthlyReport(year, month);

        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);
        BigDecimal totalSales = calculateTotalSales(startDate, endDate);

        int orderCount = calculateOrderCountFromHistoricalData(startDate, endDate, totalSales);
        BigDecimal avgOrderAmount = orderCount > 0 ?
            totalSales.divide(BigDecimal.valueOf(orderCount), 2, RoundingMode.HALF_UP) : defaultAvgOrderAmount;
        BigDecimal grossProfit = calcGrossProfit(totalSales, grossProfitRateMonthly);

        report.setMetrics(totalSales, orderCount, avgOrderAmount, grossProfit);
        report.setCategoryBreakdown(calculateCategoryBreakdown(startDate, endDate));
        report.setProductRanking(calculateProductRanking(startDate, endDate));
        report.setChannelBreakdown(calculateChannelBreakdown(startDate, endDate));

        return report;
    }

    private int calculateOrderCountFromHistoricalData(LocalDateTime start, LocalDateTime end, BigDecimal totalSales) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        int totalOrderCount = reports.stream()
                .filter(r -> r.getTotalOrderCount() != null && r.getTotalOrderCount() > 0)
                .mapToInt(SalesReport::getTotalOrderCount)
                .sum();
        if (totalOrderCount == 0 && totalSales.compareTo(BigDecimal.ZERO) > 0) {
            totalOrderCount = Math.max(totalSales.divide(defaultAvgOrderAmount, 0, RoundingMode.DOWN).intValue(), 1);
        }
        return totalOrderCount;
    }

    private BigDecimal calculateTotalSales(LocalDateTime start, LocalDateTime end) {
        return repository.findByDateRange(start, end).stream()
                .map(SalesReport::getTotalSalesAmount)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private String calculateCategoryBreakdown(LocalDateTime start, LocalDateTime end) {
        Map<String, BigDecimal> categorySales = aggregateSalesByField(start, end, SalesReport::getCategoryBreakdown);
        if (categorySales.isEmpty()) return "{}";
        return toPercentageJson(categorySales);
    }

    private String calculateProductRanking(LocalDateTime start, LocalDateTime end) {
        Map<String, BigDecimal> productSales = aggregateSalesByField(start, end, SalesReport::getProductRanking);
        if (productSales.isEmpty()) return "[]";
        return productSales.entrySet().stream()
                .sorted(Map.Entry.<String, BigDecimal>comparingByValue().reversed())
                .limit(PRODUCT_RANKING_LIMIT)
                .map(this::toProductItem)
                .collect(Collectors.toList())
                .toString();
    }

    private String calculateChannelBreakdown(LocalDateTime start, LocalDateTime end) {
        Map<String, BigDecimal> channelSales = aggregateSalesByField(start, end, SalesReport::getChannelBreakdown);
        if (channelSales.isEmpty()) return "{}";
        return toPercentageJson(channelSales);
    }

    private Map<String, BigDecimal> aggregateSalesByField(LocalDateTime start, LocalDateTime end,
            java.util.function.Function<SalesReport, String> fieldFn) {
        Map<String, BigDecimal> sales = new HashMap<>();
        for (SalesReport r : repository.findByDateRange(start, end)) {
            String field = fieldFn.apply(r);
            if (field != null && !field.isEmpty()) {
                sales.merge(field, r.getTotalSalesAmount() != null ? r.getTotalSalesAmount() : BigDecimal.ZERO, BigDecimal::add);
            }
        }
        return sales;
    }

    private String toPercentageJson(Map<String, BigDecimal> sales) {
        BigDecimal total = sales.values().stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        if (total.compareTo(BigDecimal.ZERO) <= 0) return "{}";
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, BigDecimal> entry : sales.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            BigDecimal pct = entry.getValue().multiply(BigDecimal.valueOf(100)).divide(total, 1, RoundingMode.HALF_UP);
            sb.append("\"").append(entry.getKey()).append("\":\"").append(pct).append("%\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private Map<String, Object> toProductItem(Map.Entry<String, BigDecimal> entry) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("name", entry.getKey());
        item.put("sales", entry.getValue());
        return item;
    }

    private static BigDecimal calcGrossProfit(BigDecimal totalSales, BigDecimal rate) {
        return totalSales.multiply(rate).setScale(2, RoundingMode.HALF_UP);
    }

    private static BigDecimal calcProfitMargin(BigDecimal grossProfit, BigDecimal totalSales, int orderCount) {
        if (orderCount <= 0) return BigDecimal.ZERO;
        return grossProfit.divide(totalSales.multiply(BigDecimal.valueOf(orderCount)), 4, RoundingMode.HALF_UP)
                .multiply(BigDecimal.valueOf(100));
    }
}
