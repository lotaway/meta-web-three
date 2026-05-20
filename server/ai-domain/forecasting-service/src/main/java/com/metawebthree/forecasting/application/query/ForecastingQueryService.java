package com.metawebthree.forecasting.application.query;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Service
public class ForecastingQueryService {

    private final SalesForecastRepository forecastRepository;
    private final ForecastModelRepository modelRepository;

    public ForecastingQueryService(
            SalesForecastRepository forecastRepository,
            ForecastModelRepository modelRepository) {
        this.forecastRepository = forecastRepository;
        this.modelRepository = modelRepository;
    }

    public Optional<SalesForecast> getForecastById(Long id) {
        return forecastRepository.findById(id);
    }

    public List<SalesForecast> getForecastBySkuCode(String skuCode) {
        return forecastRepository.findBySkuCode(skuCode);
    }

    public List<SalesForecast> getForecastByWarehouse(Long warehouseId) {
        return forecastRepository.findByWarehouseId(warehouseId);
    }

    public List<SalesForecast> getForecastByDate(LocalDate forecastDate) {
        return forecastRepository.findByForecastDate(forecastDate);
    }

    public List<SalesForecast> getForecastHistory(String skuCode, 
            LocalDate startDate, LocalDate endDate) {
        return forecastRepository.findBySkuCodeAndForecastDateBetween(
            skuCode, startDate, endDate);
    }

    public List<SalesForecast> getForecastByStatus(SalesForecast.ForecastStatus status) {
        return forecastRepository.findByStatus(status);
    }

    public Optional<ForecastModel> getModelById(Long id) {
        return modelRepository.findById(id);
    }

    public Optional<ForecastModel> getModelByName(String modelName) {
        return modelRepository.findByModelName(modelName);
    }

    public List<ForecastModel> getAllModels() {
        return modelRepository.findAll();
    }

    public List<ForecastModel> getModelsByStatus(ForecastModel.ModelStatus status) {
        return modelRepository.findByStatus(status);
    }

    public Optional<ForecastModel> getDeployedModel(String modelType) {
        return modelRepository.findByModelTypeAndStatus(
            modelType, ForecastModel.ModelStatus.DEPLOYED);
    }
}