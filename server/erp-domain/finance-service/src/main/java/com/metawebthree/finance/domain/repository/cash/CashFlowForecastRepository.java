package com.metawebthree.finance.domain.repository.cash;

import com.metawebthree.finance.domain.entity.cash.CashFlowForecast;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface CashFlowForecastRepository {
    CashFlowForecast save(CashFlowForecast forecast);
    Optional<CashFlowForecast> findById(Long id);
    Optional<CashFlowForecast> findByForecastNo(String forecastNo);
    List<CashFlowForecast> findAll();
    List<CashFlowForecast> findByForecastDate(LocalDate forecastDate);
    List<CashFlowForecast> findByDateRange(LocalDate startDate, LocalDate endDate);
    void deleteById(Long id);
}