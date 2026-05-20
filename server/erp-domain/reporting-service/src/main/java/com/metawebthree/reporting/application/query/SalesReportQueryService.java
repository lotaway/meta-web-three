package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class SalesReportQueryService {
    private final SalesReportRepository repository;

    public SalesReportQueryService(SalesReportRepository repository) {
        this.repository = repository;
    }

    public Long generateDailyReport() {
        LocalDateTime now = LocalDateTime.now();
        SalesReport report = new SalesReport();
        report.generateDailyReport(now);
        
        BigDecimal totalSales = calculateTotalSales(now.toLocalDate().atStartOfDay(), now.toLocalDate().atTime(23, 59, 59));
        report.setMetrics(totalSales, 150, totalSales.multiply(BigDecimal.valueOf(0.25)), BigDecimal.valueOf(25));
        report.setCategoryBreakdown("{\"electronics\":40%,\"clothing\":30%,\"food\":20%,\"others\":10%}");
        report.setChannelBreakdown("{\"online\":60%,\"offline\":30%,\"wholesale\":10%}");
        
        repository.save(report);
        return report.getId();
    }

    public Long generateMonthlyReport(int year, int month) {
        SalesReport report = new SalesReport();
        report.generateMonthlyReport(year, month);
        
        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);
        BigDecimal totalSales = calculateTotalSales(startDate, endDate);
        
        report.setMetrics(totalSales, 4500, 
            totalSales.divide(BigDecimal.valueOf(4500), 2, BigDecimal.ROUND_HALF_UP),
            totalSales.multiply(BigDecimal.valueOf(0.28)), BigDecimal.valueOf(28));
        report.setCategoryBreakdown(getCategoryBreakdown(year, month));
        report.setProductRanking(getProductRanking(year, month));
        report.setChannelBreakdown("{\"online\":55%,\"offline\":35%,\"wholesale\":10%}");
        
        repository.save(report);
        return report.getId();
    }

    private BigDecimal calculateTotalSales(LocalDateTime start, LocalDateTime end) {
        return BigDecimal.valueOf(Math.random() * 1000000 + 500000).setScale(2, BigDecimal.ROUND_HALF_UP);
    }

    private String getCategoryBreakdown(int year, int month) {
        return "{\"electronics\":38%,\"clothing\":28%,\"food\":22%,\"others\":12%}";
    }

    private String getProductRanking(int year, int month) {
        return "[{\"rank\":1,\"name\":\"Product A\",\"sales\":125000},{\"rank\":2,\"name\":\"Product B\",\"sales\":98000},{\"rank\":3,\"name\":\"Product C\",\"sales\":87000}]";
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