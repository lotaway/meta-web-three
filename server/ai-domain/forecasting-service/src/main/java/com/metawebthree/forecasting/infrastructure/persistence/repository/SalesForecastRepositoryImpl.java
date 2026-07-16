package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public class SalesForecastRepositoryImpl implements SalesForecastRepository {

    private final SalesForecastJpaRepository jpaRepository;

    public SalesForecastRepositoryImpl(SalesForecastJpaRepository jpaRepository) {
        this.jpaRepository = jpaRepository;
    }

    @Override
    public Optional<SalesForecast> findById(Long id) {
        return jpaRepository.findById(id);
    }

    @Override
    public List<SalesForecast> findBySkuCode(String skuCode) {
        return jpaRepository.findBySkuCode(skuCode);
    }

    @Override
    public List<SalesForecast> findByWarehouseId(Long warehouseId) {
        return jpaRepository.findByWarehouseId(warehouseId);
    }

    @Override
    public List<SalesForecast> findByForecastDate(LocalDate forecastDate) {
        return jpaRepository.findByForecastDate(forecastDate);
    }

    @Override
    public List<SalesForecast> findBySkuCodeAndForecastDateBetween(
            String skuCode, LocalDate startDate, LocalDate endDate) {
        return jpaRepository.findBySkuCodeAndForecastDateBetween(skuCode, startDate, endDate);
    }

    @Override
    public List<SalesForecast> findByStatus(SalesForecast.ForecastStatus status) {
        return jpaRepository.findByStatus(status);
    }

    @Override
    public SalesForecast save(SalesForecast forecast) {
        return jpaRepository.save(forecast);
    }

    @Override
    public void update(SalesForecast forecast) {
        jpaRepository.save(forecast);
    }

    @Override
    public void deleteById(Long id) {
        jpaRepository.deleteById(id);
    }
}
