package com.metawebthree.finance.domain.entity.ledger;

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
public class GeneralLedger {
    private Long id;
    private String ledgerNo;
    private Integer periodYear;
    private Integer periodMonth;
    private LedgerStatus status;
    private BigDecimal totalDebit;
    private BigDecimal totalCredit;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<GeneralLedgerEntry> entries;

    public enum LedgerStatus {
        DRAFT, POSTED, CLOSED
    }

    public static GeneralLedger create(String ledgerNo, Integer periodYear, Integer periodMonth, String createdBy) {
        LocalDateTime now = LocalDateTime.now();
        return GeneralLedger.builder()
                .ledgerNo(ledgerNo)
                .periodYear(periodYear)
                .periodMonth(periodMonth)
                .status(LedgerStatus.DRAFT)
                .totalDebit(BigDecimal.ZERO)
                .totalCredit(BigDecimal.ZERO)
                .createdBy(createdBy)
                .entries(new ArrayList<>())
                .createdAt(now)
                .updatedAt(now)
                .build();
    }

    public void addEntry(Long subjectId, String subjectCode, String subjectName,
                         BigDecimal debitAmount, BigDecimal creditAmount) {
        if (entries == null) {
            entries = new ArrayList<>();
        }
        GeneralLedgerEntry entry = GeneralLedgerEntry.builder()
                .subjectId(subjectId)
                .subjectCode(subjectCode)
                .subjectName(subjectName)
                .debitAmount(debitAmount)
                .creditAmount(creditAmount)
                .build();
        entries.add(entry);
        recalculateTotals();
    }

    public void post() {
        if (status != LedgerStatus.DRAFT) {
            throw new IllegalStateException("Only draft ledger can be posted");
        }
        if (!isBalanced()) {
            throw new IllegalStateException("Ledger is not balanced");
        }
        this.status = LedgerStatus.POSTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void close() {
        if (status != LedgerStatus.POSTED) {
            throw new IllegalStateException("Only posted ledger can be closed");
        }
        this.status = LedgerStatus.CLOSED;
        this.updatedAt = LocalDateTime.now();
    }

    private void recalculateTotals() {
        totalDebit = entries.stream()
                .map(e -> e.getDebitAmount() != null ? e.getDebitAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        totalCredit = entries.stream()
                .map(e -> e.getCreditAmount() != null ? e.getCreditAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    public boolean isBalanced() {
        return totalDebit.compareTo(totalCredit) == 0;
    }

    @Getter
    @Setter
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GeneralLedgerEntry {
        private Long id;
        private Long subjectId;
        private String subjectCode;
        private String subjectName;
        private BigDecimal debitAmount;
        private BigDecimal creditAmount;
        private String voucherNo;
        private LocalDateTime voucherDate;
        private String description;
    }
}