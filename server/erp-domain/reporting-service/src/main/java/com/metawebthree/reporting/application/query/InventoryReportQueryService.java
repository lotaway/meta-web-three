package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.InventoryReport;
import com.metawebthree.reporting.domain.repository.InventoryReportRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class InventoryReportQueryService {
    private final InventoryReportRepository repository;

    public InventoryReportQueryService(InventoryReportRepository repository) {
        this.repository = repository;
    }

    public Long generateDailyReport() {
        LocalDateTime now = LocalDateTime.now();
        InventoryReport report = new InventoryReport();
        report.generateDailyReport(now);
        
        report.setMetrics(
            BigDecimal.valueOf(5000000),
            2500,
            80000,
            BigDecimal.valueOf(4.5),
            BigDecimal.valueOf(8.2),
            205
        );
        report.setWarehouseBreakdown("{\"WH01\":40%,\"WH02\":35%,\"WH03\":25%}");
        report.setCategoryBreakdown("{\"electronics\":45%,\"clothing\":25%,\"food\":20%,\"others\":10%}");
        report.setLowStockItems("[{\"sku\":\"SKU001\",\"name\":\"Product A\",\"stock\":50},{\"sku\":\"SKU002\",\"name\":\"Product B\",\"stock\":30}]");
        
        repository.save(report);
        return report.getId();
    }

    public Long generateMonthlyReport(int year, int month) {
        InventoryReport report = new InventoryReport();
        report.generateMonthlyReport(year, month);
        
        report.setMetrics(
            BigDecimal.valueOf(5200000),
            2600,
            82000,
            BigDecimal.valueOf(4.8),
            BigDecimal.valueOf(7.5),
            195
        );
        report.setWarehouseBreakdown("{\"WH01\":42%,\"WH02\":33%,\"WH03\":25%}");
        report.setCategoryBreakdown("{\"electronics\":43%,\"clothing\":27%,\"food\":18%,\"others\":12%}");
        
        repository.save(report);
        return report.getId();
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
}