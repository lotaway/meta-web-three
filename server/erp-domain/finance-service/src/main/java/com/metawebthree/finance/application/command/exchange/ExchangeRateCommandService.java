package com.metawebthree.finance.application.command.exchange;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import com.metawebthree.finance.domain.repository.exchange.ExchangeRateRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ExchangeRateCommandService {

    private final ExchangeRateRepository exchangeRateRepository;

    @Transactional
    public ExchangeRate createRate(String sourceCurrency, String targetCurrency,
            java.math.BigDecimal rate, LocalDate effectiveDate,
            ExchangeRate.ExchangeRateType rateType, String createdBy) {
        ExchangeRate exchangeRate = ExchangeRate.create(
                sourceCurrency, targetCurrency, rate, effectiveDate, rateType, createdBy);
        return exchangeRateRepository.save(exchangeRate);
    }

    @Transactional
    public ExchangeRate updateRate(Long id, java.math.BigDecimal newRate) {
        ExchangeRate exchangeRate = exchangeRateRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Exchange rate not found"));
        exchangeRate.updateRate(newRate);
        return exchangeRateRepository.save(exchangeRate);
    }

    @Transactional
    public void deactivateRate(Long id) {
        ExchangeRate exchangeRate = exchangeRateRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Exchange rate not found"));
        exchangeRate.deactivate();
        exchangeRateRepository.save(exchangeRate);
    }

    @Transactional
    public void deleteRate(Long id) {
        exchangeRateRepository.delete(id);
    }

    public List<ExchangeRate> getActiveRates() {
        return exchangeRateRepository.findActiveRates();
    }
}