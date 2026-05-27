package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class SalesReportQueryService {
    
    private static final BigDecimal DEFAULT_AVG_ORDER_AMOUNT = new BigDecimal("500.00");
    private static final BigDecimal GROSS_PROFIT_RATE_DAILY = new BigDecimal("0.25");
    private static final BigDecimal GROSS_PROFIT_RATE_MONTHLY = new BigDecimal("0.28");
    
    private final SalesReportRepository repository;

    public SalesReportQueryService(SalesReportRepository repository) {
        this.repository = repository;
    }

    public Long generateDailyReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);
        
        SalesReport report = new SalesReport();
        report.generateDailyReport(now);
        
        BigDecimal totalSales = calculateTotalSales(dayStart, dayEnd);
        int orderCount = calculateOrderCountFromHistoricalData(dayStart, dayEnd, totalSales);
        BigDecimal grossProfit = totalSales.multiply(GROSS_PROFIT_RATE_DAILY).setScale(2, RoundingMode.HALF_UP);
        BigDecimal profitMargin = orderCount > 0 ? 
            grossProfit.divide(totalSales.multiply(BigDecimal.valueOf(orderCount)), 4, RoundingMode.HALF_UP).multiply(BigDecimal.valueOf(100)) 
            : BigDecimal.ZERO;
        
        report.setMetrics(totalSales, orderCount, grossProfit, profitMargin);
        report.setCategoryBreakdown(calculateCategoryBreakdown(dayStart, dayEnd));
        report.setChannelBreakdown(calculateChannelBreakdown(dayStart, dayEnd));
        
        repository.save(report);
        return report.getId();
    }

    public Long generateMonthlyReport(int year, int month) {
        SalesReport report = new SalesReport();
        report.generateMonthlyReport(year, month);
        
        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);
        BigDecimal totalSales = calculateTotalSales(startDate, endDate);
        
        int orderCount = calculateOrderCountFromHistoricalData(startDate, endDate, totalSales);
        BigDecimal avgOrderAmount = orderCount > 0 ? 
            totalSales.divide(BigDecimal.valueOf(orderCount), 2, RoundingMode.HALF_UP) : DEFAULT_AVG_ORDER_AMOUNT;
        BigDecimal grossProfit = totalSales.multiply(GROSS_PROFIT_RATE_MONTHLY).setScale(2, RoundingMode.HALF_UP);
        
        report.setMetrics(totalSales, orderCount, avgOrderAmount, grossProfit);
        report.setCategoryBreakdown(calculateCategoryBreakdown(startDate, endDate));
        report.setProductRanking(calculateProductRanking(startDate, endDate));
        report.setChannelBreakdown(calculateChannelBreakdown(startDate, endDate));
        
        repository.save(report);
        return report.getId();
    }

    private int calculateOrderCountFromHistoricalData(LocalDateTime start, LocalDateTime end, BigDecimal totalSales) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        
        int totalOrderCount = 0;
        for (SalesReport r : reports) {
            if (r.getTotalOrderCount() != null && r.getTotalOrderCount() > 0) {
                totalOrderCount += r.getTotalOrderCount();
            }
        }
        
        if (totalOrderCount == 0 && totalSales.compareTo(BigDecimal.ZERO) > 0) {
            totalOrderCount = totalSales.divide(DEFAULT_AVG_ORDER_AMOUNT, 0, RoundingMode.DOWN).intValue();
            totalOrderCount = Math.max(totalOrderCount, 1);
        }
        
        return totalOrderCount;
    }

    private BigDecimal calculateTotalSales(LocalDateTime start, LocalDateTime end) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        return reports.stream()
            .map(SalesReport::getTotalSalesAmount)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private String calculateCategoryBreakdown(LocalDateTime start, LocalDateTime end) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        
        Map<String, BigDecimal> categorySales = new HashMap<>();
        for (SalesReport r : reports) {
            String category = r.getCategoryBreakdown();
            if (category != null && !category.isEmpty()) {
                categorySales.merge(category, r.getTotalSalesAmount(), BigDecimal::add);
            }
        }
        
        if (categorySales.isEmpty()) {
            return "{}";
        }
        
        BigDecimal total = categorySales.values().stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        Map<String, String> breakdown = new LinkedHashMap<>();
        for (Map.Entry<String, BigDecimal> entry : categorySales.entrySet()) {
            String percentage = total.compareTo(BigDecimal.ZERO) > 0 ?
                entry.getValue().multiply(BigDecimal.valueOf(100)).divide(total, 1, RoundingMode.HALF_UP).toString() + "%" : "0%";
            breakdown.put(entry.getKey(), percentage);
        }
        
        return new com.fasterxml.jackson.databind.ObjectMapper().valueToTree(breakdown).toString();
    }

    private String calculateProductRanking(LocalDateTime start, LocalDateTime end) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        
        Map<String, BigDecimal> productSales = new HashMap<>();
        for (SalesReport r : reports) {
            String ranking = r.getProductRanking();
            if (ranking != null && !ranking.isEmpty()) {
                productSales.merge("Product", r.getTotalSalesAmount(), BigDecimal::add);
            }
        }
        
        if (productSales.isEmpty()) {
            return "[]";
        }
        
        return productSales.entrySet().stream()
            .sorted(Map.Entry.<String, BigDecimal>comparingByValue().reversed())
            .limit(10)
            .map(e -> {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("name", e.getKey());
                item.put("sales", e.getValue());
                return item;
            })
            .collect(Collectors.toList())
            .toString();
    }

    private String calculateChannelBreakdown(LocalDateTime start, LocalDateTime end) {
        List<SalesReport> reports = repository.findByDateRange(start, end);
        
        Map<String, BigDecimal> channelSales = new HashMap<>();
        for (SalesReport r : reports) {
            String channel = r.getChannelBreakdown();
            if (channel != null && !channel.isEmpty()) {
                channelSales.merge(channel, r.getTotalSalesAmount(), BigDecimal::add);
            }
        }
        
        if (channelSales.isEmpty()) {
            return "{}";
        }
        
        BigDecimal total = channelSales.values().stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        Map<String, String> breakdown = new LinkedHashMap<>();
        for (Map.Entry<String, BigDecimal> entry : channelSales.entrySet()) {
            String percentage = total.compareTo(BigDecimal.ZERO) > 0 ?
                entry.getValue().multiply(BigDecimal.valueOf(100)).divide(total, 1, RoundingMode.HALF_UP).toString() + "%" : "0%";
            breakdown.put(entry.getKey(), percentage);
        }
        
        return new com.fasterxml.jackson.databind.ObjectMapper().valueToTree(breakdown).toString();
    }

    public Optional<SalesReport> getById(Long id) {
        return repository.findById(id);
    }

    public List<SalesReport> listByType(String type) {
        SalesReport.ReportType t = SalesReport.ReportType.valueOf(type.toUpperCase());
        return repository.findByType(t);
    }

    public List<SalesReport> listAll() {
        return repository.findAll();
    }
}