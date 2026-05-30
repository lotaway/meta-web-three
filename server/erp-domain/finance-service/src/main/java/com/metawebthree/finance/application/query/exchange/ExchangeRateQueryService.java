package com.metawebthree.finance.application.query.exchange;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import com.metawebthree.finance.domain.repository.exchange.ExchangeRateRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ExchangeRateQueryService {

    private final ExchangeRateRepository exchangeRateRepository;

    public ExchangeRate getById(Long id) {
        return exchangeRateRepository.findById(id).orElse(null);
    }

    public List<ExchangeRate> getRatesByCurrencyPair(String sourceCurrency, String targetCurrency) {
        return exchangeRateRepository.findBySourceAndTargetCurrency(sourceCurrency, targetCurrency);
    }

    public ExchangeRate getEffectiveRate(String sourceCurrency, String targetCurrency, LocalDate date) {
        return exchangeRateRepository.findEffectiveRate(sourceCurrency, targetCurrency, date).orElse(null);
    }

    public List<ExchangeRate> getActiveRates() {
        return exchangeRateRepository.findActiveRates();
    }
}