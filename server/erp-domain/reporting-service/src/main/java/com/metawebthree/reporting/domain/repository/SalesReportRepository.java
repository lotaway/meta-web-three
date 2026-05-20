package com.metawebthree.reporting.domain.repository;

import com.metawebthree.reporting.domain.entity.SalesReport;
import java.util.List;
import java.util.Optional;

public interface SalesReportRepository {
    Optional<SalesReport> findById(Long id);
    Optional<SalesReport> findByReportNo(String reportNo);
    List<SalesReport> findByType(SalesReport.ReportType type);
    List<SalesReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<SalesReport> findAll();
    void save(SalesReport report);
    void update(SalesReport report);
    void delete(Long id);
}