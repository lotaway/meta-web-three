package com.metawebthree.finance.domain.service;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import com.metawebthree.finance.domain.repository.exchange.ExchangeRateRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class ExchangeGainLossService {

    private final ExchangeRateRepository exchangeRateRepository;

    public BigDecimal calculateGainLoss(BigDecimal originalAmount, BigDecimal originalRate,
            BigDecimal currentRate) {
        if (originalAmount == null || originalRate == null || currentRate == null) {
            return BigDecimal.ZERO;
        }
        BigDecimal originalBaseAmount = originalAmount.multiply(originalRate)
                .setScale(2, RoundingMode.HALF_UP);
        BigDecimal currentBaseAmount = originalAmount.multiply(currentRate)
                .setScale(2, RoundingMode.HALF_UP);
        return currentBaseAmount.subtract(originalBaseAmount);
    }

    public BigDecimal convertToBaseCurrency(BigDecimal foreignAmount, BigDecimal exchangeRate) {
        if (foreignAmount == null || exchangeRate == null) {
            return BigDecimal.ZERO;
        }
        return foreignAmount.multiply(exchangeRate).setScale(2, RoundingMode.HALF_UP);
    }

    public BigDecimal convertFromBaseCurrency(BigDecimal baseAmount, BigDecimal exchangeRate) {
        if (baseAmount == null || exchangeRate == null || exchangeRate.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return baseAmount.divide(exchangeRate, 2, RoundingMode.HALF_UP);
    }

    public Optional<ExchangeRate> getEffectiveRate(String sourceCurrency, String targetCurrency,
            LocalDate date) {
        return exchangeRateRepository.findEffectiveRate(sourceCurrency, targetCurrency, date);
    }

    public boolean isMultiCurrencyVoucher(String currency) {
        return currency != null && !currency.isEmpty() && !currency.equalsIgnoreCase("CNY");
    }
}