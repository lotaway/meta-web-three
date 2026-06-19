package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import com.metawebthree.finance.domain.repository.VoucherRepository;
import com.metawebthree.finance.domain.repository.ledger.GeneralLedgerRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class FinancialReportQueryService {
    private final AccountRepository accountRepository;
    private final AccountSubjectRepository subjectRepository;
    private final GeneralLedgerRepository ledgerRepository;
    private final VoucherRepository voucherRepository;

    public FinancialReportQueryService(AccountRepository accountRepository, 
            AccountSubjectRepository subjectRepository,
            GeneralLedgerRepository ledgerRepository,
            VoucherRepository voucherRepository) {
        this.accountRepository = accountRepository;
        this.subjectRepository = subjectRepository;
        this.ledgerRepository = ledgerRepository;
        this.voucherRepository = voucherRepository;
    }

    public Map<String, Object> getBalanceSheet(LocalDateTime asOfDate) {
        Map<String, Object> report = new HashMap<>();
        
        List<Account> assets = accountRepository.findByType(Account.AccountType.BANK);
        BigDecimal totalAssets = assets.stream()
            .filter(a -> a.getStatus() == Account.AccountStatus.ACTIVE)
            .map(Account::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        List<Account> liabilities = accountRepository.findByType(Account.AccountType.CREDIT);
        BigDecimal totalLiabilities = liabilities.stream()
            .filter(a -> a.getStatus() == Account.AccountStatus.ACTIVE)
            .map(Account::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal equity = totalAssets.subtract(totalLiabilities);
        
        report.put("reportDate", asOfDate);
        report.put("totalAssets", totalAssets);
        report.put("totalLiabilities", totalLiabilities);
        report.put("totalEquity", equity);
        report.put("assets", assets);
        report.put("liabilities", liabilities);
        
        return report;
    }

    public Map<String, Object> getIncomeStatement(LocalDateTime startDate, LocalDateTime endDate) {
        Map<String, Object> report = new HashMap<>();
        
        List<AccountSubject> revenueSubjects = subjectRepository.findBySubjectCodeLike("4%");
        BigDecimal totalRevenue = revenueSubjects.stream()
            .map(AccountSubject::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        List<AccountSubject> expenseSubjects = subjectRepository.findBySubjectCodeLike("5%");
        BigDecimal totalExpenses = expenseSubjects.stream()
            .map(AccountSubject::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal netProfit = totalRevenue.subtract(totalExpenses);
        
        report.put("startDate", startDate);
        report.put("endDate", endDate);
        report.put("totalRevenue", totalRevenue);
        report.put("totalExpenses", totalExpenses);
        report.put("netProfit", netProfit);
        report.put("revenueSubjects", revenueSubjects);
        report.put("expenseSubjects", expenseSubjects);
        
        return report;
    }

    public Map<String, Object> getTrialBalance(LocalDateTime asOfDate) {
        Map<String, Object> report = new HashMap<>();
        
        List<AccountSubject> allSubjects = subjectRepository.findAll();
        
        BigDecimal totalDebit = allSubjects.stream()
            .filter(s -> s.isDebitDirection())
            .map(AccountSubject::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal totalCredit = allSubjects.stream()
            .filter(s -> !s.isDebitDirection())
            .map(AccountSubject::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        report.put("reportDate", asOfDate);
        report.put("totalDebit", totalDebit);
        report.put("totalCredit", totalCredit);
        report.put("isBalanced", totalDebit.compareTo(totalCredit) == 0);
        report.put("subjects", allSubjects);
        
        return report;
    }
    
    public Map<String, Object> getCashFlowStatement(LocalDateTime startDate, LocalDateTime endDate) {
        List<GeneralLedger> ledgers = getLedgersForPeriod(startDate, endDate);
        Map<String, Object> report = buildCashFlowReport(ledgers, startDate, endDate);
        return report;
    }
    
    private List<GeneralLedger> getLedgersForPeriod(LocalDateTime start, LocalDateTime end) {
        return ledgerRepository.findByPeriodBetween(
            start.getYear(), start.getMonthValue(), end.getYear(), end.getMonthValue());
    }
    
    private Map<String, Object> buildCashFlowReport(List<GeneralLedger> ledgers, 
            LocalDateTime startDate, LocalDateTime endDate) {
        Map<String, Object> report = new HashMap<>();
        BigDecimal[] cashFlows = calculateAllCashFlows(ledgers);
        BigDecimal beginCash = getCashBalanceBeforeDate(startDate);
        
        report.put("startDate", startDate);
        report.put("endDate", endDate);
        report.put("operatingCashFlow", cashFlows[0]);
        report.put("investingCashFlow", cashFlows[1]);
        report.put("financingCashFlow", cashFlows[2]);
        report.put("netCashChange", cashFlows[3]);
        report.put("beginningCash", beginCash);
        report.put("endingCash", beginCash.add(cashFlows[3]));
        report.put("operatingDetails", getOperatingDetails(ledgers));
        report.put("investingDetails", getInvestingDetails(ledgers));
        report.put("financingDetails", getFinancingDetails(ledgers));
        return report;
    }
    
    private BigDecimal[] calculateAllCashFlows(List<GeneralLedger> ledgers) {
        BigDecimal operating = calculateOperatingCashFlow(ledgers);
        BigDecimal investing = calculateInvestingCashFlow(ledgers);
        BigDecimal financing = calculateFinancingCashFlow(ledgers);
        return new BigDecimal[]{operating, investing, financing, 
            operating.add(investing).add(financing)};
    }
    
    private BigDecimal calculateOperatingCashFlow(List<GeneralLedger> ledgers) {
        // Operating activities: codes 1xxx (cash received) - 4xxx (cash paid)
        // Simplified calculation based on revenue and expense accounts
        BigDecimal cashIn = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && e.getSubjectCode().startsWith("4"))
            .map(GeneralLedger.GeneralLedgerEntry::getCreditAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal cashOut = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && (e.getSubjectCode().startsWith("5") || e.getSubjectCode().startsWith("6")))
            .map(GeneralLedger.GeneralLedgerEntry::getDebitAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        return cashIn.subtract(cashOut);
    }
    
    private BigDecimal calculateInvestingCashFlow(List<GeneralLedger> ledgers) {
        // Investing activities: fixed assets, long-term investments
        BigDecimal cashIn = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && 
                (e.getSubjectCode().startsWith("15") || e.getSubjectCode().startsWith("16")))
            .map(GeneralLedger.GeneralLedgerEntry::getCreditAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal cashOut = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && 
                (e.getSubjectCode().startsWith("15") || e.getSubjectCode().startsWith("16")))
            .map(GeneralLedger.GeneralLedgerEntry::getDebitAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        return cashIn.subtract(cashOut);
    }
    
    private BigDecimal calculateFinancingCashFlow(List<GeneralLedger> ledgers) {
        // Financing activities: borrowings (2xxx), capital contributions (4001, 4002)
        BigDecimal cashIn = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && 
                (e.getSubjectCode().startsWith("2") || e.getSubjectCode().matches("^400[12]")))
            .map(GeneralLedger.GeneralLedgerEntry::getCreditAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal cashOut = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && 
                (e.getSubjectCode().startsWith("2") || e.getSubjectCode().matches("^400[12]")))
            .map(GeneralLedger.GeneralLedgerEntry::getDebitAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        return cashIn.subtract(cashOut);
    }
    
    private BigDecimal getCashBalanceBeforeDate(LocalDateTime date) {
        // Get cash account balances before the given date using Voucher
        LocalDateTime startOfTime = LocalDateTime.of(2020, 1, 1, 0, 0);
        List<Voucher> vouchers = voucherRepository.findByVoucherDateBetween(startOfTime, date);
        return vouchers.stream()
            .flatMap(v -> v.getLines() != null ? v.getLines().stream() : java.util.stream.Stream.empty())
            .filter(l -> {
                if (l.getSubjectId() == null) return false;
                return subjectRepository.findById(l.getSubjectId())
                    .map(s -> s.getSubjectCode() != null && s.getSubjectCode().startsWith("1001"))
                    .orElse(false);
            })
            .map(l -> {
                BigDecimal debit = l.getDebitAmount() != null ? l.getDebitAmount() : BigDecimal.ZERO;
                BigDecimal credit = l.getCreditAmount() != null ? l.getCreditAmount() : BigDecimal.ZERO;
                return debit.subtract(credit);
            })
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
    
    private List<Map<String, Object>> getOperatingDetails(List<GeneralLedger> ledgers) {
        List<Map<String, Object>> details = new ArrayList<>();
        details.add(buildDetailItem("Cash received from sales", 
            calculateCashBySubjectCode(ledgers, "4001", true)));
        details.add(buildDetailItem("Cash paid to suppliers", 
            calculateCashBySubjectCode(ledgers, "5001", false)));
        details.add(buildDetailItem("Cash paid to employees", 
            calculateCashBySubjectCode(ledgers, "5002", false)));
        return details;
    }
    
    private Map<String, Object> buildDetailItem(String item, BigDecimal amount) {
        Map<String, Object> detail = new HashMap<>();
        detail.put("item", item);
        detail.put("amount", amount);
        return detail;
    }
    
    private BigDecimal calculateCashBySubjectCode(List<GeneralLedger> ledgers, 
            String subjectCode, boolean isCredit) {
        return ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() 
                : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null 
                && e.getSubjectCode().startsWith(subjectCode))
            .map(e -> isCredit ? e.getCreditAmount() : e.getDebitAmount())
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
    
    private List<Map<String, Object>> getInvestingDetails(List<GeneralLedger> ledgers) {
        List<Map<String, Object>> details = new ArrayList<>();
        
        // Fixed asset purchases
        Map<String, Object> fa = new HashMap<>();
        fa.put("item", "Fixed asset purchases");
        BigDecimal faCash = ledgers.stream()
            .flatMap(l -> l.getEntries() != null ? l.getEntries().stream() : java.util.stream.Stream.empty())
            .filter(e -> e.getSubjectCode() != null && e.getSubjectCode().startsWith("1601"))
            .map(GeneralLedger.GeneralLedgerEntry::getDebitAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        fa.put("amount", faCash);
        details.add(fa);
        
        return details;
    }
    
    private List<Map<String, Object>> getFinancingDetails(List<GeneralLedger> ledgers) {
        List<Map<String, Object>> details = new ArrayList<>();
        details.add(buildDetailItem("Borrowings received", 
            calculateCashBySubjectCode(ledgers, "2001", true)));
        details.add(buildDetailItem("Capital contributions", 
            calculateCashBySubjectCode(ledgers, "4003", true)));
        return details;
    }
}