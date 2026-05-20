package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class FinancialReportQueryService {
    private final AccountRepository accountRepository;
    private final AccountSubjectRepository subjectRepository;

    public FinancialReportQueryService(AccountRepository accountRepository, 
            AccountSubjectRepository subjectRepository) {
        this.accountRepository = accountRepository;
        this.subjectRepository = subjectRepository;
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
}