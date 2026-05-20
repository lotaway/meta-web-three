package com.metawebthree.finance.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class Voucher {
    private Long id;
    private String voucherNo;
    private VoucherType type;
    private LocalDateTime voucherDate;
    private String description;
    private VoucherStatus status;
    private String createdBy;
    private String approvedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<VoucherLine> lines;

    public enum VoucherType {
        RECEIPT, PAYMENT, TRANSFER, GENERAL
    }

    public enum VoucherStatus {
        DRAFT, PENDING_APPROVAL, APPROVED, POSTED, REJECTED
    }

    public void createDraft(String voucherNo, VoucherType type, String description, String createdBy) {
        this.voucherNo = voucherNo;
        this.type = type;
        this.voucherDate = LocalDateTime.now();
        this.description = description;
        this.status = VoucherStatus.DRAFT;
        this.createdBy = createdBy;
        this.lines = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addLine(Long subjectId, BigDecimal debitAmount, BigDecimal creditAmount) {
        VoucherLine line = new VoucherLine();
        line.subjectId = subjectId;
        line.debitAmount = debitAmount;
        line.creditAmount = creditAmount;
        lines.add(line);
    }

    public void submitForApproval() {
        validateForApproval();
        status = VoucherStatus.PENDING_APPROVAL;
        updatedAt = LocalDateTime.now();
    }

    public void approve(String approver) {
        if (status != VoucherStatus.PENDING_APPROVAL) {
            throw new IllegalStateException("Voucher not pending approval");
        }
        approvedBy = approver;
        status = VoucherStatus.APPROVED;
        updatedAt = LocalDateTime.now();
    }

    public void reject(String approver, String reason) {
        if (status != VoucherStatus.PENDING_APPROVAL) {
            throw new IllegalStateException("Voucher not pending approval");
        }
        approvedBy = approver;
        description = description + " [REJECTED: " + reason + "]";
        status = VoucherStatus.REJECTED;
        updatedAt = LocalDateTime.now();
    }

    public void post() {
        if (status != VoucherStatus.APPROVED) {
            throw new IllegalStateException("Voucher not approved");
        }
        if (!isBalanced()) {
            throw new IllegalStateException("Voucher not balanced");
        }
        status = VoucherStatus.POSTED;
        updatedAt = LocalDateTime.now();
    }

    private void validateForApproval() {
        if (status != VoucherStatus.DRAFT) {
            throw new IllegalStateException("Only draft vouchers can be submitted");
        }
        if (lines == null || lines.isEmpty()) {
            throw new IllegalStateException("Voucher must have lines");
        }
    }

    public boolean isBalanced() {
        BigDecimal totalDebit = lines.stream()
            .map(l -> l.debitAmount != null ? l.debitAmount : BigDecimal.ZERO)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        BigDecimal totalCredit = lines.stream()
            .map(l -> l.creditAmount != null ? l.creditAmount : BigDecimal.ZERO)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        return totalDebit.compareTo(totalCredit) == 0;
    }

    public static class VoucherLine {
        public Long subjectId;
        public BigDecimal debitAmount;
        public BigDecimal creditAmount;
    }

    public Long getId() { return id; }
    public String getVoucherNo() { return voucherNo; }
    public VoucherType getType() { return type; }
    public LocalDateTime getVoucherDate() { return voucherDate; }
    public String getDescription() { return description; }
    public VoucherStatus getStatus() { return status; }
    public String getCreatedBy() { return createdBy; }
    public String getApprovedBy() { return approvedBy; }
    public List<VoucherLine> getLines() { return lines; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
}