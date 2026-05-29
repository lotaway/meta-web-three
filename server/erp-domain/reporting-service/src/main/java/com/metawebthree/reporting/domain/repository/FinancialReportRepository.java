package com.metawebthree.reporting.domain.repository;

import com.metawebthree.reporting.domain.entity.FinancialReport;
import java.util.List;
import java.util.Optional;

public interface FinancialReportRepository {
    Optional<FinancialReport> findById(Long id);
    Optional<FinancialReport> findByReportNo(String reportNo);
    List<FinancialReport> findByType(FinancialReport.ReportType type);
    List<FinancialReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<FinancialReport> findAll();
    void save(FinancialReport report);
    void update(FinancialReport report);
    void delete(Long id);
}