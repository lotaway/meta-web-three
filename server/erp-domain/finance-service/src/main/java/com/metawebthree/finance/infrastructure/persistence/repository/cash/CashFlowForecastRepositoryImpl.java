package com.metawebthree.finance.infrastructure.persistence.repository.cash;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cash.CashFlowForecast;
import com.metawebthree.finance.domain.repository.cash.CashFlowForecastRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 现金流预测仓储（本地开发用内存实现）。
 */
@Repository
public class CashFlowForecastRepositoryImpl implements CashFlowForecastRepository {

    private final Map<Long, CashFlowForecast> forecasts = new ConcurrentHashMap<>();
    private final Map<Long, CashFlowForecast.ForecastItem> items = new ConcurrentHashMap<>();

    @Override
    public Long save(CashFlowForecast forecast) {
        if (forecast.getId() == null) {
            forecast.setId(IdWorker.getId());
        }
        forecasts.put(forecast.getId(), forecast);
        return forecast.getId();
    }

    @Override
    public Optional<CashFlowForecast> findById(Long id) {
        return Optional.ofNullable(forecasts.get(id));
    }

    @Override
    public Optional<CashFlowForecast> findByForecastNo(String forecastNo) {
        return forecasts.values().stream()
                .filter(f -> Objects.equals(f.getForecastNo(), forecastNo))
                .findFirst();
    }

    @Override
    public List<CashFlowForecast> findAll() {
        return forecasts.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CashFlowForecast> findByForecastDate(LocalDate forecastDate) {
        return forecasts.values().stream()
                .filter(f -> Objects.equals(f.getForecastDate(), forecastDate))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashFlowForecast> findByDateRange(LocalDate startDate, LocalDate endDate) {
        return forecasts.values().stream()
                .filter(f -> f.getForecastDate() != null
                        && (startDate == null || !f.getForecastDate().isBefore(startDate))
                        && (endDate == null || !f.getForecastDate().isAfter(endDate)))
                .collect(Collectors.toList());
    }

    @Override
    public Long saveItem(CashFlowForecast.ForecastItem item) {
        if (item.getId() == null) {
            item.setId(IdWorker.getId());
        }
        items.put(item.getId(), item);
        return item.getId();
    }

    @Override
    public List<CashFlowForecast.ForecastItem> findItemsByForecastId(Long forecastId) {
        return items.values().stream()
                .filter(i -> Objects.equals(i.getForecastId(), forecastId))
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        forecasts.remove(id);
    }
}
