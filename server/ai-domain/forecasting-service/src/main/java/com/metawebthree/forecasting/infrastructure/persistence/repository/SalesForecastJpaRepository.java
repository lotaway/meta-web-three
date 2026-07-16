package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
public interface SalesForecastJpaRepository extends JpaRepository<SalesForecast, Long> {

    List<SalesForecast> findBySkuCode(String skuCode);

    List<SalesForecast> findByWarehouseId(Long warehouseId);

    List<SalesForecast> findByForecastDate(LocalDate forecastDate);

    List<SalesForecast> findBySkuCodeAndForecastDateBetween(
        String skuCode, LocalDate startDate, LocalDate endDate);

    List<SalesForecast> findByStatus(SalesForecast.ForecastStatus status);
}
