package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.finance.application.command.VoucherCommandService;
import com.metawebthree.finance.application.query.VoucherQueryService;
import com.metawebthree.finance.domain.entity.Voucher;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.math.BigDecimal;
import java.util.List;

@RestController
@RequestMapping("/api/finance/vouchers")
public class VoucherController {
    private final VoucherCommandService commandService;
    private final VoucherQueryService queryService;

    public VoucherController(VoucherCommandService commandService, VoucherQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping
    public ResponseEntity<Long> createVoucher(@RequestBody VoucherCreateRequest request) {
        Long id = commandService.createVoucher(request.getVoucherNo(), request.getType(), 
            request.getDescription(), request.getCreatedBy());
        return ResponseEntity.ok(id);
    }

    @PostMapping("/{id}/lines")
    public ResponseEntity<Void> addLine(@PathVariable Long id, @RequestBody VoucherLineRequest request) {
        commandService.addVoucherLine(id, request.getSubjectId(), 
            request.getDebitAmount(), request.getCreditAmount());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/submit")
    public ResponseEntity<Void> submitForApproval(@PathVariable Long id) {
        commandService.submitForApproval(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/approve")
    public ResponseEntity<Void> approve(@PathVariable Long id, @RequestParam String approver) {
        commandService.approve(id, approver);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/reject")
    public ResponseEntity<Void> reject(@PathVariable Long id, @RequestParam String approver, 
                                        @RequestParam String reason) {
        commandService.reject(id, approver, reason);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/post")
    public ResponseEntity<Void> post(@PathVariable Long id) {
        commandService.post(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}")
    public ResponseEntity<Voucher> getVoucher(@PathVariable Long id) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping
    public ResponseEntity<List<Voucher>> listVouchers(@RequestParam(required = false) String status) {
        List<Voucher> vouchers = status != null ? 
            queryService.listByStatus(status) : queryService.listAll();
        return ResponseEntity.ok(vouchers);
    }

    public static class VoucherCreateRequest {
        private String voucherNo;
        private String type;
        private String description;
        private String createdBy;

        public String getVoucherNo() { return voucherNo; }
        public void setVoucherNo(String voucherNo) { this.voucherNo = voucherNo; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCreatedBy() { return createdBy; }
        public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    }

    public static class VoucherLineRequest {
        private Long subjectId;
        private BigDecimal debitAmount;
        private BigDecimal creditAmount;

        public Long getSubjectId() { return subjectId; }
        public void setSubjectId(Long subjectId) { this.subjectId = subjectId; }
        public BigDecimal getDebitAmount() { return debitAmount; }
        public void setDebitAmount(BigDecimal debitAmount) { this.debitAmount = debitAmount; }
        public BigDecimal getCreditAmount() { return creditAmount; }
        public void setCreditAmount(BigDecimal creditAmount) { this.creditAmount = creditAmount; }
    }
}