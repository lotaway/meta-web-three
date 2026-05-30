package com.metawebthree.finance.interfaces.facade.cash;

import com.metawebthree.finance.application.command.cash.CashCommandService;
import com.metawebthree.finance.application.command.cash.dto.*;
import com.metawebthree.finance.application.query.cash.CashQueryService;
import com.metawebthree.finance.domain.entity.cash.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/cash")
public class CashController {
    private final CashCommandService commandService;
    private final CashQueryService queryService;

    public CashController(CashCommandService commandService, CashQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // ==================== Cash Plan APIs ====================
    @PostMapping("/plan")
    public ResponseEntity<Map<String, Object>> createCashPlan(@RequestBody CashPlanCreateCommand command) {
        Long id = commandService.createCashPlan(command);
        return ResponseEntity.ok(Map.of("id", id, "success", true));
    }

    @PutMapping("/plan/{id}")
    public ResponseEntity<Map<String, Object>> updateCashPlan(@PathVariable Long id, @RequestBody CashPlanCreateCommand command) {
        command.setPlanCode(id.toString());
        commandService.updateCashPlan(command);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @GetMapping("/plan/{id}")
    public ResponseEntity<CashPlan> getCashPlan(@PathVariable Long id) {
        CashPlan plan = queryService.getCashPlanById(id);
        return ResponseEntity.ok(plan);
    }

    @GetMapping("/plan/code/{code}")
    public ResponseEntity<CashPlan> getCashPlanByCode(@PathVariable String code) {
        CashPlan plan = queryService.getCashPlanByCode(code);
        return ResponseEntity.ok(plan);
    }

    @GetMapping("/plan/list")
    public ResponseEntity<List<CashPlan>> listCashPlans(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long departmentId) {
        List<CashPlan> plans;
        if (status != null) {
            plans = queryService.listCashPlansByStatus(status);
        } else if (departmentId != null) {
            plans = queryService.listCashPlansByDepartment(departmentId);
        } else {
            plans = queryService.listAllCashPlans();
        }
        return ResponseEntity.ok(plans);
    }

    @GetMapping("/plan/{id}/lines")
    public ResponseEntity<List<CashPlanLine>> getCashPlanLines(@PathVariable Long id) {
        List<CashPlanLine> lines = queryService.getCashPlanLines(id);
        return ResponseEntity.ok(lines);
    }

    @PostMapping("/plan/{id}/submit")
    public ResponseEntity<Map<String, Object>> submitCashPlan(@PathVariable Long id) {
        commandService.submitCashPlan(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/plan/{id}/approve")
    public ResponseEntity<Map<String, Object>> approveCashPlan(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveCashPlan(id, request.getApproverId(), request.getApproverName());
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/plan/{id}/reject")
    public ResponseEntity<Map<String, Object>> rejectCashPlan(@PathVariable Long id) {
        commandService.rejectCashPlan(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @DeleteMapping("/plan/{id}")
    public ResponseEntity<Map<String, Object>> deleteCashPlan(@PathVariable Long id) {
        commandService.deleteCashPlan(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    // ==================== Bank Account APIs ====================
    @PostMapping("/account")
    public ResponseEntity<Map<String, Object>> createBankAccount(@RequestBody BankAccountCreateCommand command) {
        Long id = commandService.createBankAccount(command);
        return ResponseEntity.ok(Map.of("id", id, "success", true));
    }

    @GetMapping("/account/{id}")
    public ResponseEntity<BankAccount> getBankAccount(@PathVariable Long id) {
        BankAccount account = queryService.getBankAccountById(id);
        return ResponseEntity.ok(account);
    }

    @GetMapping("/account/code/{code}")
    public ResponseEntity<BankAccount> getBankAccountByCode(@PathVariable String code) {
        BankAccount account = queryService.getBankAccountByCode(code);
        return ResponseEntity.ok(account);
    }

    @GetMapping("/account/list")
    public ResponseEntity<List<BankAccount>> listBankAccounts(
            @RequestParam(required = false) String status) {
        List<BankAccount> accounts;
        if (status != null) {
            accounts = queryService.listBankAccountsByStatus(status);
        } else {
            accounts = queryService.listAllBankAccounts();
        }
        return ResponseEntity.ok(accounts);
    }

    @GetMapping("/account/active")
    public ResponseEntity<List<BankAccount>> listActiveBankAccounts() {
        return ResponseEntity.ok(queryService.listActiveBankAccounts());
    }

    @GetMapping("/account/total-balance")
    public ResponseEntity<Map<String, Object>> getTotalCashBalance() {
        BigDecimal total = queryService.getTotalCashBalance();
        return ResponseEntity.ok(Map.of("totalBalance", total));
    }

    @PostMapping("/account/{id}/freeze")
    public ResponseEntity<Map<String, Object>> freezeBankAccount(@PathVariable Long id) {
        commandService.freezeBankAccount(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/account/{id}/unfreeze")
    public ResponseEntity<Map<String, Object>> unfreezeBankAccount(@PathVariable Long id) {
        commandService.unfreezeBankAccount(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/account/{id}/close")
    public ResponseEntity<Map<String, Object>> closeBankAccount(@PathVariable Long id) {
        commandService.closeBankAccount(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @DeleteMapping("/account/{id}")
    public ResponseEntity<Map<String, Object>> deleteBankAccount(@PathVariable Long id) {
        commandService.deleteBankAccount(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    // ==================== Cash Transfer APIs ====================
    @PostMapping("/transfer")
    public ResponseEntity<Map<String, Object>> createCashTransfer(@RequestBody CashTransferCreateCommand command) {
        Long id = commandService.createCashTransfer(command);
        return ResponseEntity.ok(Map.of("id", id, "success", true));
    }

    @GetMapping("/transfer/{id}")
    public ResponseEntity<CashTransfer> getCashTransfer(@PathVariable Long id) {
        CashTransfer transfer = queryService.getCashTransferById(id);
        return ResponseEntity.ok(transfer);
    }

    @GetMapping("/transfer/list")
    public ResponseEntity<List<CashTransfer>> listCashTransfers(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long accountId) {
        List<CashTransfer> transfers;
        if (status != null) {
            transfers = queryService.listCashTransfersByStatus(status);
        } else if (accountId != null) {
            transfers = queryService.listCashTransfersByAccount(accountId);
        } else {
            transfers = queryService.listAllCashTransfers();
        }
        return ResponseEntity.ok(transfers);
    }

    @PostMapping("/transfer/{id}/submit")
    public ResponseEntity<Map<String, Object>> submitCashTransfer(@PathVariable Long id) {
        commandService.submitCashTransfer(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/transfer/{id}/approve")
    public ResponseEntity<Map<String, Object>> approveCashTransfer(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveCashTransfer(id, request.getApproverId(), request.getApproverName());
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/transfer/{id}/reject")
    public ResponseEntity<Map<String, Object>> rejectCashTransfer(@PathVariable Long id) {
        commandService.rejectCashTransfer(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/transfer/{id}/cancel")
    public ResponseEntity<Map<String, Object>> cancelCashTransfer(@PathVariable Long id) {
        commandService.cancelCashTransfer(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @DeleteMapping("/transfer/{id}")
    public ResponseEntity<Map<String, Object>> deleteCashTransfer(@PathVariable Long id) {
        commandService.deleteCashTransfer(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    // ==================== Bank Reconciliation APIs ====================
    @PostMapping("/reconciliation")
    public ResponseEntity<Map<String, Object>> createBankReconciliation(@RequestBody BankReconciliationCreateCommand command) {
        Long id = commandService.createBankReconciliation(command);
        return ResponseEntity.ok(Map.of("id", id, "success", true));
    }

    @GetMapping("/reconciliation/{id}")
    public ResponseEntity<BankReconciliation> getBankReconciliation(@PathVariable Long id) {
        BankReconciliation reconciliation = queryService.getBankReconciliationById(id);
        return ResponseEntity.ok(reconciliation);
    }

    @GetMapping("/reconciliation/list")
    public ResponseEntity<List<BankReconciliation>> listBankReconciliations(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long accountId) {
        List<BankReconciliation> reconciliations;
        if (status != null) {
            reconciliations = queryService.listBankReconciliationsByStatus(status);
        } else if (accountId != null) {
            reconciliations = queryService.listBankReconciliationsByAccount(accountId);
        } else {
            reconciliations = queryService.listAllBankReconciliations();
        }
        return ResponseEntity.ok(reconciliations);
    }

    @PostMapping("/reconciliation/{id}/submit")
    public ResponseEntity<Map<String, Object>> submitBankReconciliation(@PathVariable Long id) {
        commandService.submitBankReconciliation(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    @PostMapping("/reconciliation/{id}/approve")
    public ResponseEntity<Map<String, Object>> approveBankReconciliation(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveBankReconciliation(id, request.getApproverId(), request.getApproverName());
        return ResponseEntity.ok(Map.of("success", true));
    }

    @DeleteMapping("/reconciliation/{id}")
    public ResponseEntity<Map<String, Object>> deleteBankReconciliation(@PathVariable Long id) {
        commandService.deleteBankReconciliation(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    // ==================== Cash Flow Forecast APIs ====================
    @PostMapping("/forecast")
    public ResponseEntity<Map<String, Object>> createCashFlowForecast(@RequestBody CashFlowForecastCreateCommand command) {
        Long id = commandService.createCashFlowForecast(command);
        return ResponseEntity.ok(Map.of("id", id, "success", true));
    }

    @GetMapping("/forecast/{id}")
    public ResponseEntity<CashFlowForecast> getCashFlowForecast(@PathVariable Long id) {
        CashFlowForecast forecast = queryService.getCashFlowForecastById(id);
        return ResponseEntity.ok(forecast);
    }

    @GetMapping("/forecast/list")
    public ResponseEntity<List<CashFlowForecast>> listCashFlowForecasts(
            @RequestParam(required = false) String forecastDate) {
        List<CashFlowForecast> forecasts;
        if (forecastDate != null) {
            forecasts = queryService.listCashFlowForecastsByDate(java.time.LocalDate.parse(forecastDate));
        } else {
            forecasts = queryService.listAllCashFlowForecasts();
        }
        return ResponseEntity.ok(forecasts);
    }

    @GetMapping("/forecast/{id}/items")
    public ResponseEntity<List<CashFlowForecast.ForecastItem>> getCashFlowForecastItems(@PathVariable Long id) {
        List<CashFlowForecast.ForecastItem> items = queryService.getCashFlowForecastItems(id);
        return ResponseEntity.ok(items);
    }

    @DeleteMapping("/forecast/{id}")
    public ResponseEntity<Map<String, Object>> deleteCashFlowForecast(@PathVariable Long id) {
        commandService.deleteCashFlowForecast(id);
        return ResponseEntity.ok(Map.of("success", true));
    }

    // ==================== Dashboard API ====================
    @GetMapping("/summary")
    public ResponseEntity<CashQueryService.CashSummary> getCashSummary() {
        return ResponseEntity.ok(queryService.getCashSummary());
    }

    // ==================== Request DTOs ====================
    public static class ApproveRequest {
        private Long approverId;
        private String approverName;

        public Long getApproverId() { return approverId; }
        public void setApproverId(Long approverId) { this.approverId = approverId; }
        public String getApproverName() { return approverName; }
        public void setApproverName(String approverName) { this.approverName = approverName; }
    }
}