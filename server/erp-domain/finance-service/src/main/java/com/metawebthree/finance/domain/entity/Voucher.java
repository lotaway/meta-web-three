package com.metawebthree.finance.domain.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@Builder(toBuilder = true)
@NoArgsConstructor
@AllArgsConstructor
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

    public static Voucher createDraft(String voucherNo, VoucherType type, String description, String createdBy) {
        LocalDateTime now = LocalDateTime.now();
        return Voucher.builder()
                .voucherNo(voucherNo)
                .type(type)
                .voucherDate(now)
                .description(description)
                .status(VoucherStatus.DRAFT)
                .createdBy(createdBy)
                .lines(new ArrayList<>())
                .createdAt(now)
                .updatedAt(now)
                .build();
    }

    public void addLine(Long subjectId, BigDecimal debitAmount, BigDecimal creditAmount) {
        if (lines == null) {
            lines = new ArrayList<>();
        }
        VoucherLine line = VoucherLine.builder()
                .subjectId(subjectId)
                .debitAmount(debitAmount)
                .creditAmount(creditAmount)
                .build();
        lines.add(line);
    }

    public void submitForApproval() {
        validateForApproval();
        this.status = VoucherStatus.PENDING_APPROVAL;
        this.updatedAt = LocalDateTime.now();
    }

    public void approve(String approver) {
        if (status != VoucherStatus.PENDING_APPROVAL) {
            throw new IllegalStateException("Voucher not pending approval");
        }
        this.approvedBy = approver;
        this.status = VoucherStatus.APPROVED;
        this.updatedAt = LocalDateTime.now();
    }

    public void reject(String approver, String reason) {
        if (status != VoucherStatus.PENDING_APPROVAL) {
            throw new IllegalStateException("Voucher not pending approval");
        }
        this.approvedBy = approver;
        this.description = this.description + " [REJECTED: " + reason + "]";
        this.status = VoucherStatus.REJECTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void post() {
        if (status != VoucherStatus.APPROVED) {
            throw new IllegalStateException("Voucher not approved");
        }
        if (!isBalanced()) {
            throw new IllegalStateException("Voucher not balanced");
        }
        this.status = VoucherStatus.POSTED;
        this.updatedAt = LocalDateTime.now();
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

    @Getter
    @Setter
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class VoucherLine {
        private Long subjectId;
        private BigDecimal debitAmount;
        private BigDecimal creditAmount;
    }
}
