package com.metawebthree.finance.domain.service;

import com.metawebthree.finance.domain.entity.FinancialRatio;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import com.metawebthree.finance.domain.repository.FinancialRatioRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.Map;

@Service
public class FinancialRatioDomainService {
    private static final int DAYS_IN_YEAR = 365;
    private final FinancialRatioRepository ratioRepository;
    private final AccountRepository accountRepository;
    private final AccountSubjectRepository subjectRepository;

    public FinancialRatioDomainService(
            FinancialRatioRepository ratioRepository,
            AccountRepository accountRepository,
            AccountSubjectRepository subjectRepository) {
        this.ratioRepository = ratioRepository;
        this.accountRepository = accountRepository;
        this.subjectRepository = subjectRepository;
    }

    public BigDecimal calculateInventoryTurnover(BigDecimal costOfGoodsSold, BigDecimal averageInventory) {
        if (averageInventory == null || averageInventory.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return costOfGoodsSold.divide(averageInventory, 4, RoundingMode.HALF_UP);
    }

    public BigDecimal calculateAccountsReceivableTurnoverDays(BigDecimal creditSales, BigDecimal averageReceivables) {
        if (averageReceivables == null || averageReceivables.compareTo(BigDecimal.ZERO) == 0
                || creditSales == null || creditSales.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.valueOf(-1);
        }
        BigDecimal turnoverRate = creditSales.divide(averageReceivables, 4, RoundingMode.HALF_UP);
        if (turnoverRate.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.valueOf(-1);
        }
        return BigDecimal.valueOf(DAYS_IN_YEAR).divide(turnoverRate, 2, RoundingMode.HALF_UP);
    }

    public BigDecimal calculateAccountsPayableTurnoverDays(BigDecimal creditPurchases, BigDecimal averagePayables) {
        if (averagePayables == null || averagePayables.compareTo(BigDecimal.ZERO) == 0
                || creditPurchases == null || creditPurchases.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.valueOf(-1);
        }
        BigDecimal turnoverRate = creditPurchases.divide(averagePayables, 4, RoundingMode.HALF_UP);
        if (turnoverRate.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.valueOf(-1);
        }
        return BigDecimal.valueOf(DAYS_IN_YEAR).divide(turnoverRate, 2, RoundingMode.HALF_UP);
    }

    public BigDecimal calculateGrossMargin(BigDecimal revenue, BigDecimal costOfGoodsSold) {
        if (revenue == null || revenue.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return revenue.subtract(costOfGoodsSold)
                .divide(revenue, 4, RoundingMode.HALF_UP)
                .multiply(BigDecimal.valueOf(100));
    }

    public FinancialRatio calculateAndSaveInventoryTurnover(BigDecimal costOfGoodsSold, BigDecimal averageInventory, String period) {
        BigDecimal value = calculateInventoryTurnover(costOfGoodsSold, averageInventory);
        return saveRatio("INVENTORY_TURNOVER", value, period);
    }

    public FinancialRatio calculateAndSaveReceivableTurnoverDays(BigDecimal creditSales, BigDecimal averageReceivables, String period) {
        BigDecimal value = calculateAccountsReceivableTurnoverDays(creditSales, averageReceivables);
        return saveRatio("ACCOUNTS_RECEIVABLE_TURNOVER_DAYS", value, period);
    }

    public FinancialRatio calculateAndSavePayableTurnoverDays(BigDecimal creditPurchases, BigDecimal averagePayables, String period) {
        BigDecimal value = calculateAccountsPayableTurnoverDays(creditPurchases, averagePayables);
        return saveRatio("ACCOUNTS_PAYABLE_TURNOVER_DAYS", value, period);
    }

    public FinancialRatio calculateAndSaveGrossMargin(BigDecimal revenue, BigDecimal costOfGoodsSold, String period) {
        BigDecimal value = calculateGrossMargin(revenue, costOfGoodsSold);
        return saveRatio("GROSS_MARGIN", value, period);
    }

    private FinancialRatio saveRatio(String ratioType, BigDecimal value, String period) {
        FinancialRatio ratio = new FinancialRatio();
        ratio.setRatioType(ratioType);
        ratio.setValue(value);
        ratio.setPeriod(period);
        ratio.setCalculatedAt(LocalDateTime.now());
        ratioRepository.save(ratio);
        return ratio;
    }

    public Map<String, BigDecimal> calculateAllRatios(BigDecimal revenue, BigDecimal costOfGoodsSold,
            BigDecimal averageInventory, BigDecimal averageReceivables, BigDecimal averagePayables) {
        return Map.of(
                "inventoryTurnover", calculateInventoryTurnover(costOfGoodsSold, averageInventory),
                "receivableTurnoverDays", calculateAccountsReceivableTurnoverDays(revenue, averageReceivables),
                "payableTurnoverDays", calculateAccountsPayableTurnoverDays(costOfGoodsSold, averagePayables),
                "grossMargin", calculateGrossMargin(revenue, costOfGoodsSold)
        );
    }
}