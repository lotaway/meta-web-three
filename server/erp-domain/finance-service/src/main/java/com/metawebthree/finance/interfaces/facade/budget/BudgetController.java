package com.metawebthree.finance.interfaces.facade.budget;

import com.metawebthree.finance.application.command.budget.BudgetCommandService;
import com.metawebthree.finance.application.command.budget.dto.BudgetAdjustmentCreateCommand;
import com.metawebthree.finance.application.command.budget.dto.BudgetCreateCommand;
import com.metawebthree.finance.application.command.budget.dto.BudgetUpdateCommand;
import com.metawebthree.finance.application.query.budget.BudgetQueryService;
import com.metawebthree.finance.domain.entity.budget.Budget;
import com.metawebthree.finance.domain.entity.budget.BudgetAdjustment;
import com.metawebthree.finance.domain.entity.budget.BudgetLine;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/budget")
public class BudgetController {
    private final BudgetCommandService commandService;
    private final BudgetQueryService queryService;

    public BudgetController(BudgetCommandService commandService, BudgetQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping
    public ResponseEntity<IdResponse> createBudget(@RequestBody BudgetCreateCommand command) {
        Long id = commandService.createBudget(command);
        return ResponseEntity.ok(new IdResponse(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<Void> updateBudget(@PathVariable Long id, @RequestBody BudgetUpdateCommand command) {
        command.setId(id);
        commandService.updateBudget(command);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}")
    public ResponseEntity<Budget> getBudget(@PathVariable Long id) {
        Budget budget = queryService.getById(id);
        return ResponseEntity.ok(budget);
    }

    @GetMapping("/code/{code}")
    public ResponseEntity<Budget> getBudgetByCode(@PathVariable String code) {
        Budget budget = queryService.getByCode(code);
        return ResponseEntity.ok(budget);
    }

    @GetMapping("/list")
    public ResponseEntity<List<Budget>> listBudgets(
            @RequestParam(required = false) Long departmentId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String period) {
        List<Budget> budgets;
        if (departmentId != null) {
            budgets = queryService.listByDepartment(departmentId);
        } else if (status != null) {
            budgets = queryService.listByStatus(status);
        } else if (period != null) {
            budgets = queryService.listByPeriod(period);
        } else {
            budgets = queryService.listAll();
        }
        return ResponseEntity.ok(budgets);
    }

    @GetMapping("/{id}/lines")
    public ResponseEntity<List<BudgetLine>> getBudgetLines(@PathVariable Long id) {
        List<BudgetLine> lines = queryService.getBudgetLines(id);
        return ResponseEntity.ok(lines);
    }

    @PostMapping("/{id}/submit")
    public ResponseEntity<Void> submitBudget(@PathVariable Long id) {
        commandService.submitBudget(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/approve")
    public ResponseEntity<Void> approveBudget(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveBudget(id, request.getApproverId(), request.getApproverName());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/reject")
    public ResponseEntity<Void> rejectBudget(@PathVariable Long id) {
        commandService.rejectBudget(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/close")
    public ResponseEntity<Void> closeBudget(@PathVariable Long id) {
        commandService.closeBudget(id);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteBudget(@PathVariable Long id) {
        commandService.deleteBudget(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/adjustment")
    public ResponseEntity<IdResponse> applyAdjustment(@RequestBody BudgetAdjustmentCreateCommand command) {
        Long id = commandService.applyAdjustment(command);
        return ResponseEntity.ok(new IdResponse(id));
    }

    @GetMapping("/adjustment/pending")
    public ResponseEntity<List<BudgetAdjustment>> getPendingAdjustments() {
        List<BudgetAdjustment> adjustments = queryService.getPendingAdjustments();
        return ResponseEntity.ok(adjustments);
    }

    @GetMapping("/{id}/adjustments")
    public ResponseEntity<List<BudgetAdjustment>> getAdjustments(@PathVariable Long id) {
        List<BudgetAdjustment> adjustments = queryService.getAdjustments(id);
        return ResponseEntity.ok(adjustments);
    }

    @PostMapping("/adjustment/{id}/approve")
    public ResponseEntity<Void> approveAdjustment(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveAdjustment(id, request.getApproverId(), request.getApproverName(), request.getComment());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/adjustment/{id}/reject")
    public ResponseEntity<Void> rejectAdjustment(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.rejectAdjustment(id, request.getApproverId(), request.getApproverName(), request.getComment());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}/analysis")
    public ResponseEntity<BudgetQueryService.BudgetAnalysisResult> analyzeBudget(@PathVariable Long id) {
        BudgetQueryService.BudgetAnalysisResult result = queryService.analyzeBudget(id);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/{id}/compare")
    public ResponseEntity<BudgetQueryService.BudgetComparisonResult> compareBudget(
            @PathVariable Long id,
            @RequestParam java.math.BigDecimal actualAmount) {
        BudgetQueryService.BudgetComparisonResult result = queryService.compareBudgetWithActual(id, actualAmount);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/{id}/usage")
    public ResponseEntity<Void> recordUsage(
            @PathVariable Long id,
            @RequestBody UsageRecordRequest request) {
        commandService.recordUsage(id, request.getSubjectCode(), request.getAmount());
        return ResponseEntity.ok().build();
    }

    public record IdResponse(Long id) {}

    public static class ApproveRequest {
        private Long approverId;
        private String approverName;
        private String comment;

        public Long getApproverId() { return approverId; }
        public String getApproverName() { return approverName; }
        public String getComment() { return comment; }

        public void setApproverId(Long approverId) { this.approverId = approverId; }
        public void setApproverName(String approverName) { this.approverName = approverName; }
        public void setComment(String comment) { this.comment = comment; }
    }

    public static class UsageRecordRequest {
        private String subjectCode;
        private java.math.BigDecimal amount;

        public String getSubjectCode() { return subjectCode; }
        public java.math.BigDecimal getAmount() { return amount; }

        public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
        public void setAmount(java.math.BigDecimal amount) { this.amount = amount; }
    }
}