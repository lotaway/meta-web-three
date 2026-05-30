package com.metawebthree.finance.domain.repository.exchange;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface ExchangeRateRepository {
    Optional<ExchangeRate> findById(Long id);

    List<ExchangeRate> findBySourceAndTargetCurrency(String sourceCurrency, String targetCurrency);

    Optional<ExchangeRate> findEffectiveRate(String sourceCurrency, String targetCurrency, LocalDate date);

    List<ExchangeRate> findActiveRates();

    ExchangeRate save(ExchangeRate exchangeRate);

    void delete(Long id);
}