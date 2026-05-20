package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.FinancialReport;
import com.metawebthree.reporting.domain.repository.FinancialReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class FinancialReportQueryService {
    private final FinancialReportRepository repository;

    public FinancialReportQueryService(FinancialReportRepository repository) {
        this.repository = repository;
    }

    public Long generateReceivableReport() {
        LocalDateTime now = LocalDateTime.now();
        FinancialReport report = new FinancialReport();
        report.generateReceivableReport(now);
        
        report.setReceivableMetrics(
            BigDecimal.valueOf(2500000),
            BigDecimal.valueOf(1800000)
        );
        report.setReceivablesByCustomer("{\"Customer A\":350000,\"Customer B\":280000,\"Customer C\":220000,\"Others\":450000}");
        
        repository.save(report);
        return report.getId();
    }

    public Long generatePayableReport() {
        LocalDateTime now = LocalDateTime.now();
        FinancialReport report = new FinancialReport();
        report.generatePayableReport(now);
        
        report.setReceivableMetrics(
            BigDecimal.valueOf(1200000),
            BigDecimal.valueOf(950000)
        );
        report.setPayablesBySupplier("{\"Supplier X\":450000,\"Supplier Y\":320000,\"Supplier Z\":180000}");
        
        repository.save(report);
        return report.getId();
    }

    public Long generateAgingReport() {
        LocalDateTime now = LocalDateTime.now();
        FinancialReport report = new FinancialReport();
        report.generateAgingReport(now);
        
        report.setReceivableMetrics(
            BigDecimal.valueOf(2500000),
            BigDecimal.valueOf(1800000)
        );
        report.setAgingAnalysis("{\"0-30 days\":{\"amount\":1200000,\"ratio\":48%},\"31-60 days\":{\"amount\":650000,\"ratio\":26%},\"61-90 days\":{\"amount\":400000,\"ratio\":16%},\"over 90 days\":{\"amount\":250000,\"ratio\":10%}}");
        
        repository.save(report);
        return report.getId();
    }

    public Long generateWorkingCapitalReport() {
        LocalDateTime now = LocalDateTime.now();
        FinancialReport report = new FinancialReport();
        report.generateAgingReport(now);
        
        report.setWorkingCapitalMetrics(
            BigDecimal.valueOf(5000000),
            BigDecimal.valueOf(3200000)
        );
        
        repository.save(report);
        return report.getId();
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
}