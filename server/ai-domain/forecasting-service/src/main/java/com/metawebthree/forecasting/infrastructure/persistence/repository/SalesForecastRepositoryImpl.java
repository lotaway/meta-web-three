package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import com.metawebthree.forecasting.infrastructure.persistence.mapper.SalesForecastMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public class SalesForecastRepositoryImpl implements SalesForecastRepository {

    private final SalesForecastMapper mapper;

    public SalesForecastRepositoryImpl(SalesForecastMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<SalesForecast> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<SalesForecast> findBySkuCode(String skuCode) {
        return mapper.selectList(new QueryWrapper<SalesForecast>().eq("sku_code", skuCode));
    }

    @Override
    public List<SalesForecast> findByWarehouseId(Long warehouseId) {
        return mapper.selectList(new QueryWrapper<SalesForecast>().eq("warehouse_id", warehouseId));
    }

    @Override
    public List<SalesForecast> findByForecastDate(LocalDate forecastDate) {
        return mapper.selectList(new QueryWrapper<SalesForecast>().eq("forecast_date", forecastDate));
    }

    @Override
    public List<SalesForecast> findBySkuCodeAndForecastDateBetween(
            String skuCode, LocalDate startDate, LocalDate endDate) {
        return mapper.selectList(new QueryWrapper<SalesForecast>()
            .eq("sku_code", skuCode)
            .between("forecast_date", startDate, endDate));
    }

    @Override
    public List<SalesForecast> findByStatus(SalesForecast.ForecastStatus status) {
        return mapper.selectList(new QueryWrapper<SalesForecast>().eq("status", status));
    }

    @Override
    public void save(SalesForecast forecast) {
        mapper.insert(forecast);
    }

    @Override
    public void update(SalesForecast forecast) {
        mapper.updateById(forecast);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
