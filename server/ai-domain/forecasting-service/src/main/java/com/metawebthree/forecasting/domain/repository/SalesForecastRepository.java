package com.metawebthree.forecasting.domain.repository;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface SalesForecastRepository {
    Optional<SalesForecast> findById(Long id);
    List<SalesForecast> findBySkuCode(String skuCode);
    List<SalesForecast> findByWarehouseId(Long warehouseId);
    List<SalesForecast> findByForecastDate(LocalDate forecastDate);
    List<SalesForecast> findBySkuCodeAndForecastDateBetween(
        String skuCode, LocalDate startDate, LocalDate endDate);
    List<SalesForecast> findByStatus(SalesForecast.ForecastStatus status);
    void save(SalesForecast forecast);
    void update(SalesForecast forecast);
    void deleteById(Long id);
}