package com.metawebthree.reporting.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.ERPPermissions;
import com.metawebthree.reporting.application.query.*;
import com.metawebthree.reporting.domain.entity.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/reports")
public class ReportController {
    private final SalesReportQueryService salesReportService;
    private final InventoryReportQueryService inventoryReportService;
    private final FinancialReportQueryService financialReportService;

    public ReportController(SalesReportQueryService salesReportService,
                           InventoryReportQueryService inventoryReportService,
                           FinancialReportQueryService financialReportService) {
        this.salesReportService = salesReportService;
        this.inventoryReportService = inventoryReportService;
        this.financialReportService = financialReportService;
    }

    @RequirePermission(ERPPermissions.SALES_REPORT_CREATE)
    @PostMapping("/sales/daily")
    public ResponseEntity<Void> generateDailySalesReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        salesReportService.generateDailyReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.SALES_REPORT_CREATE)
    @PostMapping("/sales/monthly")
    public ResponseEntity<Void> generateMonthlySalesReport(
            @RequestParam int year, @RequestParam int month,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        salesReportService.generateMonthlyReport(year, month);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.SALES_REPORT_READ)
    @GetMapping("/sales/{id}")
    public ResponseEntity<SalesReport> getSalesReport(@PathVariable Long id) {
        return salesReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.SALES_REPORT_READ)
    @GetMapping("/sales")
    public ResponseEntity<List<SalesReport>> listSalesReports(@RequestParam(required = false) String type) {
        List<SalesReport> reports = type != null ?
            salesReportService.listByType(type) : salesReportService.listAll();
        return ResponseEntity.ok(reports);
    }

    @RequirePermission(ERPPermissions.INVENTORY_REPORT_CREATE)
    @PostMapping("/inventory/daily")
    public ResponseEntity<Void> generateDailyInventoryReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        inventoryReportService.generateDailyReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.INVENTORY_REPORT_CREATE)
    @PostMapping("/inventory/monthly")
    public ResponseEntity<Void> generateMonthlyInventoryReport(
            @RequestParam int year, @RequestParam int month,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        inventoryReportService.generateMonthlyReport(year, month);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.INVENTORY_REPORT_READ)
    @GetMapping("/inventory/{id}")
    public ResponseEntity<InventoryReport> getInventoryReport(@PathVariable Long id) {
        return inventoryReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.INVENTORY_REPORT_READ)
    @GetMapping("/inventory")
    public ResponseEntity<List<InventoryReport>> listInventoryReports(@RequestParam(required = false) String type) {
        List<InventoryReport> reports = type != null ?
            inventoryReportService.listByType(type) : inventoryReportService.listAll();
        return ResponseEntity.ok(reports);
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_CREATE)
    @PostMapping("/financial/receivable")
    public ResponseEntity<Void> generateReceivableReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        financialReportService.generateReceivableReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_CREATE)
    @PostMapping("/financial/payable")
    public ResponseEntity<Void> generatePayableReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        financialReportService.generatePayableReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_CREATE)
    @PostMapping("/financial/aging")
    public ResponseEntity<Void> generateAgingReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        financialReportService.generateAgingReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_CREATE)
    @PostMapping("/financial/working-capital")
    public ResponseEntity<Void> generateWorkingCapitalReport(
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        financialReportService.generateWorkingCapitalReport();
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/financial/{id}")
    public ResponseEntity<FinancialReport> getFinancialReport(@PathVariable Long id) {
        return financialReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.FINANCIAL_REPORT_READ)
    @GetMapping("/financial")
    public ResponseEntity<List<FinancialReport>> listFinancialReports(@RequestParam(required = false) String type) {
        List<FinancialReport> reports = type != null ?
            financialReportService.listByType(type) : financialReportService.listAll();
        return ResponseEntity.ok(reports);
    }
}