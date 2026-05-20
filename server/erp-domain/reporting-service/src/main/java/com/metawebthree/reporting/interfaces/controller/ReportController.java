package com.metawebthree.reporting.interfaces.controller;

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

    @PostMapping("/sales/daily")
    public ResponseEntity<Long> generateDailySalesReport() {
        return ResponseEntity.ok(salesReportService.generateDailyReport());
    }

    @PostMapping("/sales/monthly")
    public ResponseEntity<Long> generateMonthlySalesReport(@RequestParam int year, @RequestParam int month) {
        return ResponseEntity.ok(salesReportService.generateMonthlyReport(year, month));
    }

    @GetMapping("/sales/{id}")
    public ResponseEntity<SalesReport> getSalesReport(@PathVariable Long id) {
        return salesReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/sales")
    public ResponseEntity<List<SalesReport>> listSalesReports(@RequestParam(required = false) String type) {
        List<SalesReport> reports = type != null ?
            salesReportService.listByType(type) : salesReportService.listAll();
        return ResponseEntity.ok(reports);
    }

    @PostMapping("/inventory/daily")
    public ResponseEntity<Long> generateDailyInventoryReport() {
        return ResponseEntity.ok(inventoryReportService.generateDailyReport());
    }

    @PostMapping("/inventory/monthly")
    public ResponseEntity<Long> generateMonthlyInventoryReport(@RequestParam int year, @RequestParam int month) {
        return ResponseEntity.ok(inventoryReportService.generateMonthlyReport(year, month));
    }

    @GetMapping("/inventory/{id}")
    public ResponseEntity<InventoryReport> getInventoryReport(@PathVariable Long id) {
        return inventoryReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/inventory")
    public ResponseEntity<List<InventoryReport>> listInventoryReports(@RequestParam(required = false) String type) {
        List<InventoryReport> reports = type != null ?
            inventoryReportService.listByType(type) : inventoryReportService.listAll();
        return ResponseEntity.ok(reports);
    }

    @PostMapping("/financial/receivable")
    public ResponseEntity<Long> generateReceivableReport() {
        return ResponseEntity.ok(financialReportService.generateReceivableReport());
    }

    @PostMapping("/financial/payable")
    public ResponseEntity<Long> generatePayableReport() {
        return ResponseEntity.ok(financialReportService.generatePayableReport());
    }

    @PostMapping("/financial/aging")
    public ResponseEntity<Long> generateAgingReport() {
        return ResponseEntity.ok(financialReportService.generateAgingReport());
    }

    @PostMapping("/financial/working-capital")
    public ResponseEntity<Long> generateWorkingCapitalReport() {
        return ResponseEntity.ok(financialReportService.generateWorkingCapitalReport());
    }

    @GetMapping("/financial/{id}")
    public ResponseEntity<FinancialReport> getFinancialReport(@PathVariable Long id) {
        return financialReportService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/financial")
    public ResponseEntity<List<FinancialReport>> listFinancialReports(@RequestParam(required = false) String type) {
        List<FinancialReport> reports = type != null ?
            financialReportService.listByType(type) : financialReportService.listAll();
        return ResponseEntity.ok(reports);
    }
}