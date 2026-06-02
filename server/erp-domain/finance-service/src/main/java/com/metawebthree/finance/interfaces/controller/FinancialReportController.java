package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.ERPPermissions;
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

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/balance-sheet")
    public ResponseEntity<Map<String, Object>> getBalanceSheet(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime asOfDate,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return ResponseEntity.ok(reportService.getBalanceSheet(asOfDate));
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/income-statement")
    public ResponseEntity<Map<String, Object>> getIncomeStatement(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return ResponseEntity.ok(reportService.getIncomeStatement(startDate, endDate));
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/trial-balance")
    public ResponseEntity<Map<String, Object>> getTrialBalance(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime asOfDate,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return ResponseEntity.ok(reportService.getTrialBalance(asOfDate));
    }
    
    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/cash-flow-statement")
    public ResponseEntity<Map<String, Object>> getCashFlowStatement(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return ResponseEntity.ok(reportService.getCashFlowStatement(startDate, endDate));
    }
}