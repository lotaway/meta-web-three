package com.metawebthree.reporting.domain.repository;

import com.metawebthree.reporting.domain.entity.InventoryReport;
import java.util.List;
import java.util.Optional;

public interface InventoryReportRepository {
    Optional<InventoryReport> findById(Long id);
    Optional<InventoryReport> findByReportNo(String reportNo);
    List<InventoryReport> findByType(InventoryReport.ReportType type);
    List<InventoryReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<InventoryReport> findAll();
    void save(InventoryReport report);
    void update(InventoryReport report);
    void delete(Long id);
}