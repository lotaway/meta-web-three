package com.metawebthree.inventory.interfaces.controller.stockcheck;

import com.metawebthree.inventory.application.StockCheckApplicationService;
import com.metawebthree.inventory.application.dto.stockcheck.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/inventory/stock-check")
@RequiredArgsConstructor
@Slf4j
public class StockCheckController {

    private final StockCheckApplicationService stockCheckService;

    // Plan endpoints
    @PostMapping("/plans")
    public ResponseEntity<StockCheckPlanDTO> createPlan(@RequestBody StockCheckPlanDTO dto) {
        StockCheckPlanDTO result = stockCheckService.createPlan(dto);
        return ResponseEntity.ok(result);
    }

    @PutMapping("/plans/{id}")
    public ResponseEntity<StockCheckPlanDTO> updatePlan(@PathVariable Long id, @RequestBody StockCheckPlanDTO dto) {
        StockCheckPlanDTO result = stockCheckService.updatePlan(id, dto);
        return ResponseEntity.ok(result);
    }

    @DeleteMapping("/plans/{id}")
    public ResponseEntity<Void> deletePlan(@PathVariable Long id) {
        stockCheckService.deletePlan(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/plans/{id}/approve")
    public ResponseEntity<StockCheckPlanDTO> approvePlan(@PathVariable Long id) {
        StockCheckPlanDTO result = stockCheckService.approvePlan(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/plans/{id}/start")
    public ResponseEntity<StockCheckPlanDTO> startPlan(@PathVariable Long id) {
        StockCheckPlanDTO result = stockCheckService.startPlan(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/plans/{id}/complete")
    public ResponseEntity<StockCheckPlanDTO> completePlan(@PathVariable Long id) {
        StockCheckPlanDTO result = stockCheckService.completePlan(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/plans/{id}/cancel")
    public ResponseEntity<StockCheckPlanDTO> cancelPlan(@PathVariable Long id) {
        StockCheckPlanDTO result = stockCheckService.cancelPlan(id);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/plans/{id}")
    public ResponseEntity<StockCheckPlanDTO> getPlan(@PathVariable Long id) {
        StockCheckPlanDTO result = stockCheckService.queryPlan(id);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/plans/no/{planNo}")
    public ResponseEntity<StockCheckPlanDTO> getPlanByNo(@PathVariable String planNo) {
        StockCheckPlanDTO result = stockCheckService.queryPlanByNo(planNo);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/plans")
    public ResponseEntity<List<StockCheckPlanDTO>> listPlans(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String status) {
        List<StockCheckPlanDTO> results = stockCheckService.listPlans(warehouseId, status);
        return ResponseEntity.ok(results);
    }

    // Record endpoints
    @PostMapping("/records")
    public ResponseEntity<StockCheckRecordDTO> createRecord(@RequestBody StockCheckRecordDTO dto) {
        StockCheckRecordDTO result = stockCheckService.createRecord(dto);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/records/batch")
    public ResponseEntity<List<StockCheckRecordDTO>> batchCreateRecords(@RequestBody List<StockCheckRecordDTO> records) {
        // For batch, we process each record
        for (StockCheckRecordDTO record : records) {
            stockCheckService.createRecord(record);
        }
        return ResponseEntity.ok(records);
    }

    @PutMapping("/records/{id}")
    public ResponseEntity<StockCheckRecordDTO> updateRecord(@PathVariable Long id, @RequestBody StockCheckRecordDTO dto) {
        StockCheckRecordDTO result = stockCheckService.updateRecord(id, dto);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/records")
    public ResponseEntity<List<StockCheckRecordDTO>> listRecords(
            @RequestParam Long planId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Boolean hasDifference) {
        List<StockCheckRecordDTO> results = stockCheckService.listRecords(planId, status, hasDifference);
        return ResponseEntity.ok(results);
    }

    // Diff endpoints
    @PostMapping("/diffs/{id}/approve")
    public ResponseEntity<StockCheckDiffDTO> approveDiff(
            @PathVariable Long id,
            @RequestParam String approver,
            @RequestParam(required = false) String remark) {
        StockCheckDiffDTO result = stockCheckService.approveDiff(id, approver, remark);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/diffs/{id}/reject")
    public ResponseEntity<StockCheckDiffDTO> rejectDiff(
            @PathVariable Long id,
            @RequestParam String approver,
            @RequestParam(required = false) String remark) {
        StockCheckDiffDTO result = stockCheckService.rejectDiff(id, approver, remark);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/diffs/{id}/process")
    public ResponseEntity<StockCheckDiffDTO> processDiff(
            @PathVariable Long id,
            @RequestParam String processor,
            @RequestParam String solution,
            @RequestParam(required = false) String remark) {
        StockCheckDiffDTO result = stockCheckService.processDiff(id, processor, solution, remark);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/diffs")
    public ResponseEntity<List<StockCheckDiffDTO>> listDiffs(
            @RequestParam(required = false) Long planId,
            @RequestParam(required = false) String approvalStatus,
            @RequestParam(required = false) String processingStatus) {
        List<StockCheckDiffDTO> results = stockCheckService.listDiffs(planId, approvalStatus, processingStatus);
        return ResponseEntity.ok(results);
    }

    @GetMapping("/diffs/pending-approval")
    public ResponseEntity<List<StockCheckDiffDTO>> listPendingApprovalDiffs() {
        List<StockCheckDiffDTO> results = stockCheckService.listPendingApprovalDiffs();
        return ResponseEntity.ok(results);
    }

    // Report endpoints
    @GetMapping("/reports/{planId}")
    public ResponseEntity<StockCheckReportDTO> generateReport(@PathVariable Long planId) {
        StockCheckReportDTO result = stockCheckService.generateReport(planId);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/reports")
    public ResponseEntity<List<StockCheckReportDTO>> listReports(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {
        List<StockCheckReportDTO> results = stockCheckService.listReports(warehouseId, startDate, endDate);
        return ResponseEntity.ok(results);
    }
}