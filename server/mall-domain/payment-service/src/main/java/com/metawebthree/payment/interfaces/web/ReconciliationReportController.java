package com.metawebthree.payment.interfaces.web;

import com.metawebthree.payment.application.ReconciliationReportService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.util.Map;

@RestController
@RequestMapping("/reconciliation")
@RequiredArgsConstructor
@Slf4j
@Tag(name = "Reconciliation", description = "Financial reconciliation APIs")
public class ReconciliationReportController {

    private final ReconciliationReportService reconciliationReportService;

    @GetMapping("/report")
    @Operation(summary = "Get reconciliation report for a specific date")
    public ResponseEntity<Map<String, Object>> getReport(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date) {
        log.info("Getting reconciliation report for date: {}", date);
        Map<String, Object> report = reconciliationReportService.getReport(date);
        return ResponseEntity.ok(report);
    }

    @GetMapping("/report/today")
    @Operation(summary = "Get today's reconciliation status")
    public ResponseEntity<Map<String, Object>> getTodayStatus() {
        LocalDate yesterday = LocalDate.now().minusDays(1);
        log.info("Getting today's reconciliation status (date: {})", yesterday);
        Map<String, Object> report = reconciliationReportService.getReport(yesterday);
        return ResponseEntity.ok(report);
    }

    @GetMapping("/pending/count")
    @Operation(summary = "Get pending differences count")
    public ResponseEntity<Long> getPendingCount(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date) {
        log.info("Getting pending differences count for date: {}", date);
        long count = reconciliationReportService.getPendingCount(date);
        return ResponseEntity.ok(count);
    }
}