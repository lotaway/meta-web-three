package com.metawebthree.finance.domain.service.ledger;

import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger.GeneralLedgerEntry;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import com.metawebthree.finance.domain.repository.ledger.GeneralLedgerRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class GeneralLedgerDomainService {
    private final GeneralLedgerRepository ledgerRepository;
    private final AccountSubjectRepository subjectRepository;

    public GeneralLedgerDomainService(GeneralLedgerRepository ledgerRepository,
                                       AccountSubjectRepository subjectRepository) {
        this.ledgerRepository = ledgerRepository;
        this.subjectRepository = subjectRepository;
    }

    public GeneralLedger generateLedgerFromVoucher(Voucher voucher) {
        if (voucher.getStatus() != Voucher.VoucherStatus.POSTED) {
            throw new IllegalStateException("Only posted vouchers can be posted to ledger");
        }

        LocalDateTime voucherDate = voucher.getVoucherDate();
        Integer periodYear = voucherDate.getYear();
        Integer periodMonth = voucherDate.getMonthValue();

        GeneralLedger ledger = ledgerRepository.findByPeriod(periodYear, periodMonth)
                .orElseGet(() -> createNewLedger(periodYear, periodMonth, voucher.getCreatedBy()));

        for (Voucher.VoucherLine line : voucher.getLines()) {
            Long subjectId = line.getSubjectId();
            AccountSubject subject = subjectRepository.findById(subjectId)
                    .orElseThrow(() -> new IllegalArgumentException("Subject not found: " + subjectId));

            ledger.addEntry(
                    subjectId,
                    subject.getSubjectCode(),
                    subject.getSubjectName(),
                    line.getDebitAmount(),
                    line.getCreditAmount()
            );
        }

        return ledger;
    }

    public Map<String, BigDecimal> getSubjectBalance(String subjectCode, Integer periodYear, Integer periodMonth) {
        List<GeneralLedgerEntry> entries = ledgerRepository.findEntriesByPeriod(periodYear, periodMonth);

        List<GeneralLedgerEntry> subjectEntries = entries.stream()
                .filter(e -> e.getSubjectCode().startsWith(subjectCode))
                .collect(Collectors.toList());

        BigDecimal totalDebit = subjectEntries.stream()
                .map(e -> e.getDebitAmount() != null ? e.getDebitAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        BigDecimal totalCredit = subjectEntries.stream()
                .map(e -> e.getCreditAmount() != null ? e.getCreditAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        AccountSubject subject = subjectRepository.findBySubjectCode(subjectCode)
                .orElseThrow(() -> new IllegalArgumentException("Subject not found: " + subjectCode));

        BigDecimal balance;
        if (subject.isDebitDirection()) {
            balance = totalDebit.subtract(totalCredit);
        } else {
            balance = totalCredit.subtract(totalDebit);
        }

        return Map.of(
                "totalDebit", totalDebit,
                "totalCredit", totalCredit,
                "balance", balance
        );
    }

    public Map<String, BigDecimal> getBalanceSheetData(Integer periodYear, Integer periodMonth) {
        List<GeneralLedgerEntry> entries = ledgerRepository.findEntriesByPeriod(periodYear, periodMonth);

        BigDecimal totalAssets = calculateTotalByPrefix(entries, "1");
        BigDecimal totalLiabilities = calculateTotalByPrefix(entries, "2");
        BigDecimal totalEquity = calculateTotalByPrefix(entries, "3");
        BigDecimal totalProfit = calculateTotalByPrefix(entries, "4", "5");

        return Map.of(
                "totalAssets", totalAssets,
                "totalLiabilities", totalLiabilities,
                "totalEquity", totalEquity.add(totalProfit),
                "totalLiabilitiesAndEquity", totalLiabilities.add(totalEquity).add(totalProfit)
        );
    }

    public Map<String, BigDecimal> getIncomeStatementData(Integer periodYear, Integer periodMonth) {
        List<GeneralLedgerEntry> entries = ledgerRepository.findEntriesByPeriod(periodYear, periodMonth);

        BigDecimal totalRevenue = calculateTotalByPrefix(entries, "4");
        BigDecimal totalExpenses = calculateTotalByPrefix(entries, "5");
        BigDecimal netProfit = totalRevenue.subtract(totalExpenses);

        return Map.of(
                "totalRevenue", totalRevenue,
                "totalExpenses", totalExpenses,
                "netProfit", netProfit
        );
    }

    public Map<String, BigDecimal> getCashFlowStatementData(Integer periodYear, Integer periodMonth) {
        List<GeneralLedgerEntry> entries = ledgerRepository.findEntriesByPeriod(periodYear, periodMonth);

        BigDecimal operatingCashFlow = calculateOperatingCashFlow(entries);
        BigDecimal investingCashFlow = calculateInvestingCashFlow(entries);
        BigDecimal financingCashFlow = calculateFinancingCashFlow(entries);
        BigDecimal netCashFlow = operatingCashFlow.add(investingCashFlow).add(financingCashFlow);

        return Map.of(
                "operatingCashFlow", operatingCashFlow,
                "investingCashFlow", investingCashFlow,
                "financingCashFlow", financingCashFlow,
                "netCashFlow", netCashFlow
        );
    }

    private GeneralLedger createNewLedger(Integer periodYear, Integer periodMonth, String createdBy) {
        String ledgerNo = String.format("GL-%04d%02d", periodYear, periodMonth);
        GeneralLedger ledger = GeneralLedger.create(ledgerNo, periodYear, periodMonth, createdBy);
        ledgerRepository.save(ledger);
        return ledger;
    }

    private BigDecimal calculateTotalByPrefix(List<GeneralLedgerEntry> entries, String... prefixes) {
        BigDecimal total = BigDecimal.ZERO;
        for (String prefix : prefixes) {
            BigDecimal debit = entries.stream()
                    .filter(e -> e.getSubjectCode().startsWith(prefix))
                    .map(e -> e.getDebitAmount() != null ? e.getDebitAmount() : BigDecimal.ZERO)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            BigDecimal credit = entries.stream()
                    .filter(e -> e.getSubjectCode().startsWith(prefix))
                    .map(e -> e.getCreditAmount() != null ? e.getCreditAmount() : BigDecimal.ZERO)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            total = total.add(debit).add(credit);
        }
        return total;
    }

    private BigDecimal calculateOperatingCashFlow(List<GeneralLedgerEntry> entries) {
        BigDecimal cashInflow = entries.stream()
                .filter(e -> e.getSubjectCode().startsWith("1") || e.getSubjectCode().startsWith("4"))
                .map(e -> e.getDebitAmount() != null ? e.getDebitAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        BigDecimal cashOutflow = entries.stream()
                .filter(e -> e.getSubjectCode().startsWith("1") || e.getSubjectCode().startsWith("4"))
                .map(e -> e.getCreditAmount() != null ? e.getCreditAmount() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        return cashInflow.subtract(cashOutflow);
    }

    private BigDecimal calculateInvestingCashFlow(List<GeneralLedgerEntry> entries) {
        return entries.stream()
                .filter(e -> e.getSubjectCode().startsWith("16"))
                .map(e -> e.getCreditAmount() != null ? e.getCreditAmount().negate() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private BigDecimal calculateFinancingCashFlow(List<GeneralLedgerEntry> entries) {
        return entries.stream()
                .filter(e -> e.getSubjectCode().startsWith("2") || e.getSubjectCode().startsWith("4"))
                .map(e -> e.getCreditAmount() != null ? e.getCreditAmount().negate() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
}