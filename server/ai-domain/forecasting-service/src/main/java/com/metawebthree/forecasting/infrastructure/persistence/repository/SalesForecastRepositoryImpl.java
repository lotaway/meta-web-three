package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class SalesForecastRepositoryImpl implements SalesForecastRepository {

    private final Map<Long, SalesForecast> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public Optional<SalesForecast> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public List<SalesForecast> findBySkuCode(String skuCode) {
        return storage.values().stream()
            .filter(f -> f.getSkuCode().equals(skuCode))
            .collect(Collectors.toList());
    }

    @Override
    public List<SalesForecast> findByWarehouseId(Long warehouseId) {
        return storage.values().stream()
            .filter(f -> f.getWarehouseId().equals(warehouseId))
            .collect(Collectors.toList());
    }

    @Override
    public List<SalesForecast> findByForecastDate(LocalDate forecastDate) {
        return storage.values().stream()
            .filter(f -> f.getForecastDate().equals(forecastDate))
            .collect(Collectors.toList());
    }

    @Override
    public List<SalesForecast> findBySkuCodeAndForecastDateBetween(
            String skuCode, LocalDate startDate, LocalDate endDate) {
        return storage.values().stream()
            .filter(f -> f.getSkuCode().equals(skuCode))
            .filter(f -> !f.getForecastDate().isBefore(startDate))
            .filter(f -> !f.getForecastDate().isAfter(endDate))
            .collect(Collectors.toList());
    }

    @Override
    public List<SalesForecast> findByStatus(SalesForecast.ForecastStatus status) {
        return storage.values().stream()
            .filter(f -> f.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public SalesForecast save(SalesForecast forecast) {
        if (forecast.getId() == null) {
            forecast.setId(idGenerator.getAndIncrement());
        }
        storage.put(forecast.getId(), forecast);
        return forecast;
    }

    @Override
    public void update(SalesForecast forecast) {
        if (forecast.getId() != null && storage.containsKey(forecast.getId())) {
            storage.put(forecast.getId(), forecast);
        }
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}