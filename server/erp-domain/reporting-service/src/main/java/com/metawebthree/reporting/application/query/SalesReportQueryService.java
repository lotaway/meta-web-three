package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import com.metawebthree.reporting.domain.service.SalesReportDomainService;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class SalesReportQueryService {

    private final SalesReportDomainService domainService;
    private final SalesReportRepository repository;

    public SalesReportQueryService(SalesReportRepository repository) {
        this.domainService = new SalesReportDomainService(repository);
        this.repository = repository;
    }

    public void generateDailyReport() {
        SalesReport report = domainService.generateDailyReport(LocalDateTime.now());
        repository.save(report);
    }

    public void generateMonthlyReport(int year, int month) {
        SalesReport report = domainService.generateMonthlyReport(year, month);
        repository.save(report);
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
