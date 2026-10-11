package com.metawebthree.payment.application;

import com.metawebthree.payment.domain.repository.ReconciliationDiffRepository;
import com.metawebthree.payment.infrastructure.persistence.dataobject.ReconciliationDiffDO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class ReconciliationReportService {

    private final ReconciliationDiffRepository reconciliationDiffRepository;

    @Value("${reconciliation.report-storage-path:/tmp/reconciliation-reports}")
    private String reportStoragePath;

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter DATETIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public Map<String, Object> generateDailyReport(LocalDate reconciliationDate) {
        log.info("Generating daily reconciliation report for date: {}", reconciliationDate);

        List<ReconciliationDiffDO> allDiffs = reconciliationDiffRepository.findByReconciliationDate(reconciliationDate);
        
        Map<String, Object> report = new HashMap<>();
        report.put("reconciliationDate", reconciliationDate.format(DATE_FORMATTER));
        report.put("generatedAt", LocalDateTime.now().format(DATETIME_FORMATTER));
        
        report.put("summary", buildSummary(allDiffs));
        report.put("differences", buildDiffDetails(allDiffs));

        logReportSummary(reconciliationDate, allDiffs);
        saveCsvReport(reconciliationDate, allDiffs);

        return report;
    }

    private Map<String, Object> buildSummary(List<ReconciliationDiffDO> allDiffs) {
        long totalDiffs = allDiffs.size();
        long pendingCount = allDiffs.stream()
            .filter(d -> d.getStatus() == ReconciliationDiffDO.DiffStatus.PENDING)
            .count();
        long handledCount = allDiffs.stream()
            .filter(d -> d.getStatus() == ReconciliationDiffDO.DiffStatus.HANDLED)
            .count();
        long ignoredCount = allDiffs.stream()
            .filter(d -> d.getStatus() == ReconciliationDiffDO.DiffStatus.IGNORED)
            .count();

        long missingOrderCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.MISSING_ORDER)
            .count();
        long extraOrderCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.EXTRA_ORDER)
            .count();
        long amountMismatchCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.AMOUNT_MISMATCH)
            .count();

        BigDecimal totalAmountDiff = allDiffs.stream()
            .map(ReconciliationDiffDO::getAmountDifference)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);

        Map<String, Object> summary = new HashMap<>();
        summary.put("totalDifferences", totalDiffs);
        summary.put("pendingCount", pendingCount);
        summary.put("handledCount", handledCount);
        summary.put("ignoredCount", ignoredCount);
        summary.put("missingOrderCount", missingOrderCount);
        summary.put("extraOrderCount", extraOrderCount);
        summary.put("amountMismatchCount", amountMismatchCount);
        summary.put("totalAmountDifference", totalAmountDiff);
        
        String status = totalDiffs == 0 ? "SUCCESS" : (pendingCount > 0 ? "HAS_PENDING_ISSUES" : "ALL_HANDLED");
        summary.put("reconciliationStatus", status);
        
        return summary;
    }

    private List<Map<String, Object>> buildDiffDetails(List<ReconciliationDiffDO> allDiffs) {
        return allDiffs.stream()
            .map(this::toDiffDetail)
            .collect(Collectors.toList());
    }

    private Map<String, Object> toDiffDetail(ReconciliationDiffDO diff) {
        Map<String, Object> detail = new HashMap<>();
        detail.put("orderNo", diff.getOrderNo());
        detail.put("diffType", diff.getDiffType() != null ? diff.getDiffType().name() : "UNKNOWN");
        detail.put("internalAmount", diff.getInternalAmount());
        detail.put("externalAmount", diff.getExternalAmount());
        detail.put("amountDifference", diff.getAmountDifference());
        detail.put("status", diff.getStatus() != null ? diff.getStatus().name() : "UNKNOWN");
        detail.put("handleRemark", diff.getHandleRemark());
        detail.put("handledAt", diff.getHandledAt() != null ? diff.getHandledAt().format(DATETIME_FORMATTER) : null);
        detail.put("handledBy", diff.getHandledBy());
        return detail;
    }

    private void logReportSummary(LocalDate reconciliationDate, List<ReconciliationDiffDO> allDiffs) {
        long totalDiffs = allDiffs.size();
        long missingOrderCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.MISSING_ORDER)
            .count();
        long extraOrderCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.EXTRA_ORDER)
            .count();
        long amountMismatchCount = allDiffs.stream()
            .filter(d -> d.getDiffType() == ReconciliationDiffDO.DiffType.AMOUNT_MISMATCH)
            .count();
        BigDecimal totalAmountDiff = allDiffs.stream()
            .map(ReconciliationDiffDO::getAmountDifference)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        String status = totalDiffs == 0 ? "SUCCESS" : "HAS_ISSUES";

        log.info("=== Daily Reconciliation Report ===");
        log.info("Date: {}", reconciliationDate.format(DATE_FORMATTER));
        log.info("Total Differences: {}", totalDiffs);
        log.info("  - Missing Orders: {}", missingOrderCount);
        log.info("  - Extra Orders: {}", extraOrderCount);
        log.info("  - Amount Mismatches: {}", amountMismatchCount);
        log.info("Total Amount Difference: {}", totalAmountDiff);
        log.info("Status: {}", status);
        log.info("===================================");
    }

    private void saveCsvReport(LocalDate reconciliationDate, List<ReconciliationDiffDO> allDiffs) {
        try {
            StringBuilder csv = new StringBuilder();
            csv.append("Order No,Diff Type,Internal Amount,External Amount,Amount Difference,Status,Handle Remark,Handled At,Handled By\n");
            
            for (ReconciliationDiffDO diff : allDiffs) {
                csv.append(escapeCsv(diff.getOrderNo())).append(",");
                csv.append(escapeCsv(diff.getDiffType() != null ? diff.getDiffType().name() : "UNKNOWN")).append(",");
                csv.append(escapeCsv(formatAmount(diff.getInternalAmount()))).append(",");
                csv.append(escapeCsv(formatAmount(diff.getExternalAmount()))).append(",");
                csv.append(escapeCsv(formatAmount(diff.getAmountDifference()))).append(",");
                csv.append(escapeCsv(diff.getStatus() != null ? diff.getStatus().name() : "UNKNOWN")).append(",");
                csv.append(escapeCsv(diff.getHandleRemark())).append(",");
                csv.append(escapeCsv(diff.getHandledAt() != null ? diff.getHandledAt().format(DATETIME_FORMATTER) : "")).append(",");
                csv.append(escapeCsv(diff.getHandledBy())).append("\n");
            }

            Path dirPath = Paths.get(reportStoragePath);
            Files.createDirectories(dirPath);
            String fileName = reconciliationDate.format(DATE_FORMATTER) + ".csv";
            Path filePath = dirPath.resolve(fileName);
            Files.writeString(filePath, csv.toString());
            
            log.info("CSV Report saved to: {}", filePath);
            
        } catch (Exception e) {
            log.error("Failed to generate CSV report", e);
        }
    }

    private String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private String formatAmount(Object amount) {
        if (amount == null) {
            return "";
        }
        return amount.toString();
    }

    public Map<String, Object> getReport(LocalDate reconciliationDate) {
        return generateDailyReport(reconciliationDate);
    }

    public long getPendingCount(LocalDate reconciliationDate) {
        List<ReconciliationDiffDO> pendingDiffs = reconciliationDiffRepository.findPendingByReconciliationDate(reconciliationDate);
        return pendingDiffs.size();
    }
}