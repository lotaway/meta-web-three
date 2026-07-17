package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.entity.FinancialRatio;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import com.metawebthree.finance.domain.repository.FinancialRatioRepository;
import com.metawebthree.finance.domain.service.FinancialRatioDomainService;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class FinancialRatioQueryService {
    private final FinancialRatioRepository ratioRepository;
    private final FinancialRatioDomainService ratioDomainService;
    private final AccountRepository accountRepository;
    private final AccountSubjectRepository subjectRepository;

    public FinancialRatioQueryService(
            FinancialRatioRepository ratioRepository,
            FinancialRatioDomainService ratioDomainService,
            AccountRepository accountRepository,
            AccountSubjectRepository subjectRepository) {
        this.ratioRepository = ratioRepository;
        this.ratioDomainService = ratioDomainService;
        this.accountRepository = accountRepository;
        this.subjectRepository = subjectRepository;
    }

    public Map<String, Object> getDashboardRatios(String period) {
        Map<String, BigDecimal> ratios = calculateCurrentRatios(period);
        
        List<FinancialRatio> history = ratioRepository.findByPeriod(period);
        List<Map<String, Object>> trendData = buildTrendData(history);
        
        Map<String, Object> result = new HashMap<>();
        result.put("ratios", ratios);
        result.put("period", period);
        result.put("calculatedAt", LocalDateTime.now());
        result.put("trendData", trendData);
        result.put("status", "success");
        
        return result;
    }

    public Map<String, BigDecimal> calculateCurrentRatios(String period) {
        BigDecimal revenue = getTotalRevenue();
        BigDecimal costOfGoodsSold = getCostOfGoodsSold();
        BigDecimal averageInventory = getAverageInventory();
        BigDecimal averageReceivables = getAverageReceivables();
        BigDecimal averagePayables = getAveragePayables();
        
        return ratioDomainService.calculateAllRatios(
                revenue, costOfGoodsSold, averageInventory, averageReceivables, averagePayables
        );
    }

    private BigDecimal getTotalRevenue() {
        List<AccountSubject> revenueSubjects = subjectRepository.findBySubjectCodeLike("4%");
        return revenueSubjects.stream()
                .map(AccountSubject::getBalance)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private BigDecimal getCostOfGoodsSold() {
        List<AccountSubject> cogsSubjects = subjectRepository.findBySubjectCodeLike("5%");
        return cogsSubjects.stream()
                .map(AccountSubject::getBalance)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private BigDecimal getAverageInventory() {
        return BigDecimal.valueOf(100000);
    }

    private BigDecimal getAverageReceivables() {
        List<Account> receivables = accountRepository.findByType(Account.AccountType.CREDIT);
        BigDecimal total = receivables.stream()
                .filter(a -> a.getStatus() == Account.AccountStatus.ACTIVE)
                .map(Account::getBalance)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        List<Account> allAccounts = accountRepository.findAll();
        BigDecimal count = BigDecimal.valueOf(allAccounts.size());
        if (count.compareTo(BigDecimal.ZERO) > 0) {
            return total.divide(count, 2, RoundingMode.HALF_UP);
        }
        return total;
    }

    private BigDecimal getAveragePayables() {
        List<Account> payables = accountRepository.findByType(Account.AccountType.CREDIT);
        BigDecimal total = payables.stream()
                .filter(a -> a.getStatus() == Account.AccountStatus.ACTIVE)
                .map(Account::getBalance)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        List<Account> allAccounts = accountRepository.findAll();
        BigDecimal count = BigDecimal.valueOf(allAccounts.size());
        if (count.compareTo(BigDecimal.ZERO) > 0) {
            return total.divide(count, 2, RoundingMode.HALF_UP);
        }
        return total;
    }

    private List<Map<String, Object>> buildTrendData(List<FinancialRatio> history) {
        return history.stream()
                .sorted(Comparator.comparing(FinancialRatio::getCalculatedAt))
                .limit(12)
                .map(ratio -> {
                    Map<String, Object> item = new HashMap<>();
                    item.put("ratioType", ratio.getRatioType());
                    item.put("value", ratio.getValue());
                    item.put("calculatedAt", ratio.getCalculatedAt());
                    return item;
                })
                .collect(Collectors.toList());
    }

    public Map<String, Object> getRatioDetails(String ratioType, String period) {
        List<FinancialRatio> ratios = ratioRepository.findByRatioType(ratioType);
        
        if (!period.isEmpty()) {
            ratios = ratios.stream()
                    .filter(r -> period.equals(r.getPeriod()))
                    .collect(Collectors.toList());
        }
        
        BigDecimal current = ratios.isEmpty() ? BigDecimal.ZERO : 
                ratios.stream()
                        .map(FinancialRatio::getValue)
                        .max(Comparator.nullsLast(Comparator.naturalOrder()))
                        .orElse(BigDecimal.ZERO);
        
        BigDecimal average = ratios.isEmpty() ? BigDecimal.ZERO :
                ratios.stream()
                        .map(FinancialRatio::getValue)
                        .reduce(BigDecimal.ZERO, BigDecimal::add)
                        .divide(BigDecimal.valueOf(ratios.size()), 2, RoundingMode.HALF_UP);
        
        Map<String, Object> result = new HashMap<>();
        result.put("ratioType", ratioType);
        result.put("currentValue", current);
        result.put("averageValue", average);
        result.put("recordCount", ratios.size());
        result.put("period", period);
        
        return result;
    }

    public List<FinancialRatio> getAllRatios() {
        return ratioRepository.findAll();
    }

    public Map<String, Object> getRatioComparison(String period1, String period2) {
        List<FinancialRatio> period1Ratios = ratioRepository.findByPeriod(period1);
        List<FinancialRatio> period2Ratios = ratioRepository.findByPeriod(period2);
        
        Map<String, Object> comparison = new HashMap<>();
        
        Set<String> allTypes = new HashSet<>();
        period1Ratios.forEach(r -> allTypes.add(r.getRatioType()));
        period2Ratios.forEach(r -> allTypes.add(r.getRatioType()));
        
        for (String type : allTypes) {
            BigDecimal value1 = period1Ratios.stream()
                    .filter(r -> type.equals(r.getRatioType()))
                    .map(FinancialRatio::getValue)
                    .findFirst()
                    .orElse(BigDecimal.ZERO);
            
            BigDecimal value2 = period2Ratios.stream()
                    .filter(r -> type.equals(r.getRatioType()))
                    .map(FinancialRatio::getValue)
                    .findFirst()
                    .orElse(BigDecimal.ZERO);
            
            BigDecimal change = BigDecimal.ZERO;
            if (value1.compareTo(BigDecimal.ZERO) != 0) {
                change = value2.subtract(value1)
                        .divide(value1, 4, RoundingMode.HALF_UP)
                        .multiply(BigDecimal.valueOf(100));
            }
            
            Map<String, Object> item = new HashMap<>();
            item.put("period1", value1);
            item.put("period2", value2);
            item.put("changePercent", change);
            comparison.put(type, item);
        }
        
        return comparison;
    }
}