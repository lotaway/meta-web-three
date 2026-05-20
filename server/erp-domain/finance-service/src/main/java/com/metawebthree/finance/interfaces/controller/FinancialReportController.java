package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.finance.application.query.FinancialReportQueryService;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.time.LocalDateTime;
import java.util.Map;

@RestController
@RequestMapping("/api/finance/reports")
public class FinancialReportController {
    private final FinancialReportQueryService reportService;

    public FinancialReportController(FinancialReportQueryService reportService) {
        this.reportService = reportService;
    }

    @GetMapping("/balance-sheet")
    public ResponseEntity<Map<String, Object>> getBalanceSheet(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime asOfDate) {
        return ResponseEntity.ok(reportService.getBalanceSheet(asOfDate));
    }

    @GetMapping("/income-statement")
    public ResponseEntity<Map<String, Object>> getIncomeStatement(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        return ResponseEntity.ok(reportService.getIncomeStatement(startDate, endDate));
    }

    @GetMapping("/trial-balance")
    public ResponseEntity<Map<String, Object>> getTrialBalance(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime asOfDate) {
        return ResponseEntity.ok(reportService.getTrialBalance(asOfDate));
    }
}