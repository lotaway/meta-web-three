package com.metawebthree.forecasting.domain.service;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Service
public class ForecastingDomainServiceImpl implements ForecastingDomainService {

    private final SalesForecastRepository forecastRepository;
    private final ForecastModelRepository modelRepository;

    public ForecastingDomainServiceImpl(
            SalesForecastRepository forecastRepository,
            ForecastModelRepository modelRepository) {
        this.forecastRepository = forecastRepository;
        this.modelRepository = modelRepository;
    }

    @Override
    public SalesForecast createForecast(String skuCode, String skuName, 
            Long warehouseId, LocalDate forecastDate, Integer quantity, String modelName) {
        
        ForecastModel model = modelRepository.findByModelName(modelName)
            .orElseThrow(() -> new IllegalArgumentException("Model not found: " + modelName));
        
        if (!model.isReadyForDeployment()) {
            throw new IllegalStateException("Model is not ready for forecasting");
        }
        
        SalesForecast forecast = new SalesForecast();
        forecast.create(skuCode, skuName, warehouseId, forecastDate, quantity,
                        modelName, model.getAccuracy());
        return forecastRepository.save(forecast);
    }

    @Override
    public void confirmForecast(Long forecastId) {
        SalesForecast forecast = forecastRepository.findById(forecastId)
            .orElseThrow(() -> new IllegalArgumentException("Forecast not found"));
        forecast.confirm();
        forecastRepository.update(forecast);
    }

    @Override
    public void adjustForecast(Long forecastId, Integer newQuantity) {
        SalesForecast forecast = forecastRepository.findById(forecastId)
            .orElseThrow(() -> new IllegalArgumentException("Forecast not found"));
        forecast.adjust(newQuantity, null);
        forecastRepository.update(forecast);
    }

    @Override
    public void recordActualSales(Long forecastId, Integer actualQuantity) {
        SalesForecast forecast = forecastRepository.findById(forecastId)
            .orElseThrow(() -> new IllegalArgumentException("Forecast not found"));
        forecast.recordActual(actualQuantity, null);
        forecastRepository.update(forecast);
    }

    @Override
    public List<SalesForecast> getForecastHistory(String skuCode, LocalDate startDate,
            LocalDate endDate) {
        return forecastRepository.findBySkuCodeAndForecastDateBetween(
            skuCode, startDate, endDate);
    }

    @Override
    public ForecastModel createModel(String modelName, String modelType, 
            String algorithm, String featureConfig, Integer trainingDays) {
        ForecastModel model = new ForecastModel();
        model.create(modelName, modelType, algorithm, featureConfig, trainingDays);
        return modelRepository.save(model);
    }

    @Override
    public void trainModel(Long modelId) {
        ForecastModel model = modelRepository.findById(modelId)
            .orElseThrow(() -> new IllegalArgumentException("Model not found"));
        
        model.startTraining();
        modelRepository.update(model);
        
        // Simulate model training - in production, this would call ML training pipeline
        BigDecimal simulatedAccuracy = BigDecimal.valueOf(75 + Math.random() * 20);
        model.completeTraining(simulatedAccuracy);
        modelRepository.update(model);
    }

    @Override
    public void deployModel(Long modelId) {
        ForecastModel model = modelRepository.findById(modelId)
            .orElseThrow(() -> new IllegalArgumentException("Model not found"));
        
        if (!model.isReadyForDeployment()) {
            throw new IllegalStateException("Model accuracy below threshold (70%)");
        }
        
        model.deploy();
        modelRepository.update(model);
    }

    @Override
    public Optional<ForecastModel> getDeployedModel(String modelType) {
        return modelRepository.findByModelTypeAndStatus(
            modelType, ForecastModel.ModelStatus.DEPLOYED);
    }

    @Override
    public Double calculateForecastAccuracy(Long forecastId) {
        SalesForecast forecast = forecastRepository.findById(forecastId)
            .orElseThrow(() -> new IllegalArgumentException("Forecast not found"));
        return forecast.calculateAccuracy().doubleValue();
    }
}