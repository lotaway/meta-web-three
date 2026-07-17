package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.FinancialReport;
import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.FinancialReportRepository;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class FinancialReportQueryService {

    private static final BigDecimal COST_TO_REVENUE_RATIO = new BigDecimal("0.65");
    private static final BigDecimal RECEIVABLE_TO_REVENUE_RATIO = new BigDecimal("0.30");
    private static final BigDecimal PAYABLE_TO_COGS_RATIO = new BigDecimal("0.20");
    private static final BigDecimal CURRENT_ASSETS_TO_REVENUE_RATIO = new BigDecimal("0.6");
    private static final BigDecimal CURRENT_LIABILITIES_TO_COGS_RATIO = new BigDecimal("0.4");
    private static final BigDecimal AGING_30_DAYS_RATIO = new BigDecimal("0.48");
    private static final BigDecimal AGING_60_DAYS_RATIO = new BigDecimal("0.26");
    private static final BigDecimal AGING_90_DAYS_RATIO = new BigDecimal("0.16");
    private static final BigDecimal AGING_OVER_90_RATIO = new BigDecimal("0.10");

    private final FinancialReportRepository repository;
    private final SalesReportRepository salesReportRepository;

    public FinancialReportQueryService(FinancialReportRepository repository, SalesReportRepository salesReportRepository) {
        this.repository = repository;
        this.salesReportRepository = salesReportRepository;
    }

    public void generateReceivableReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime monthStart = now.toLocalDate().withDayOfMonth(1).atStartOfDay();
        FinMetrics metrics = calcSalesMetrics(monthStart, now);

        FinancialReport report = initReport(FinancialReport.ReportType.RECEIVABLE, now);
        report.setReceivableMetrics(metrics.totalReceivable, BigDecimal.ZERO);
        report.setReceivablesByCustomer(buildReceivablesByCustomer(metrics.totalReceivable));

        repository.save(report);
    }

    public void generatePayableReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime monthStart = now.toLocalDate().withDayOfMonth(1).atStartOfDay();
        FinMetrics metrics = calcSalesMetrics(monthStart, now);

        FinancialReport report = initReport(FinancialReport.ReportType.PAYABLE, now);
        report.setReceivableMetrics(BigDecimal.ZERO, metrics.totalPayable);
        report.setPayablesBySupplier(buildPayablesBySupplier(metrics.totalPayable));

        repository.save(report);
    }

    public void generateAgingReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime monthStart = now.toLocalDate().withDayOfMonth(1).atStartOfDay();
        FinMetrics metrics = calcSalesMetrics(monthStart, now);

        FinancialReport report = initReport(FinancialReport.ReportType.AGING, now);
        report.setReceivableMetrics(metrics.totalReceivable, metrics.totalReceivable.subtract(metrics.totalPayable));
        report.setAgingAnalysis(buildAgingAnalysis(metrics.totalReceivable));

        repository.save(report);
    }

    public void generateWorkingCapitalReport() {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime monthStart = now.toLocalDate().withDayOfMonth(1).atStartOfDay();
        FinMetrics metrics = calcSalesMetrics(monthStart, now);

        FinancialReport report = initReport(FinancialReport.ReportType.WORKING_CAPITAL, now);
        report.setWorkingCapitalMetrics(metrics.currentAssets, metrics.currentLiabilities);

        repository.save(report);
    }

    public Optional<FinancialReport> getById(Long id) {
        return repository.findById(id);
    }

    public List<FinancialReport> listByType(String type) {
        FinancialReport.ReportType t = FinancialReport.ReportType.valueOf(type.toUpperCase());
        return repository.findByType(t);
    }

    public List<FinancialReport> listAll() {
        return repository.findAll();
    }

    private FinancialReport initReport(FinancialReport.ReportType type, LocalDateTime now) {
        FinancialReport report = new FinancialReport();
        switch (type) {
            case RECEIVABLE -> report.generateReceivableReport(now);
            case PAYABLE -> report.generatePayableReport(now);
            case AGING -> report.generateAgingReport(now);
            case WORKING_CAPITAL -> report.generateWorkingCapitalReport(now);
        }
        return report;
    }

    private FinMetrics calcSalesMetrics(LocalDateTime start, LocalDateTime end) {
        BigDecimal totalSales = calcRevenue(start, end);
        BigDecimal cogs = calcCogs(totalSales);
        BigDecimal totalReceivable = calcReceivables(totalSales);
        BigDecimal totalPayable = calcPayables(cogs);
        BigDecimal currentAssets = calcCurrentAssets(totalSales);
        BigDecimal currentLiabilities = calcCurrentLiabilities(cogs);
        return new FinMetrics(totalReceivable, totalPayable, currentAssets, currentLiabilities);
    }

    private BigDecimal calcRevenue(LocalDateTime start, LocalDateTime end) {
        return salesReportRepository.findByDateRange(start, end).stream()
                .map(SalesReport::getTotalSalesAmount)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private BigDecimal calcCogs(BigDecimal totalSales) {
        return totalSales.multiply(COST_TO_REVENUE_RATIO).setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calcReceivables(BigDecimal totalSales) {
        return totalSales.multiply(RECEIVABLE_TO_REVENUE_RATIO).setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calcPayables(BigDecimal cogs) {
        return cogs.multiply(PAYABLE_TO_COGS_RATIO).setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calcCurrentAssets(BigDecimal totalSales) {
        return totalSales.multiply(CURRENT_ASSETS_TO_REVENUE_RATIO).setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calcCurrentLiabilities(BigDecimal cogs) {
        return cogs.multiply(CURRENT_LIABILITIES_TO_COGS_RATIO).setScale(2, RoundingMode.HALF_UP);
    }

    private String buildReceivablesByCustomer(BigDecimal total) {
        return "{}";
    }

    private String buildPayablesBySupplier(BigDecimal total) {
        return "{}";
    }

    private String buildAgingAnalysis(BigDecimal total) {
        return "{}";
    }

    private record FinMetrics(BigDecimal totalReceivable, BigDecimal totalPayable,
                               BigDecimal currentAssets, BigDecimal currentLiabilities) {}
}
