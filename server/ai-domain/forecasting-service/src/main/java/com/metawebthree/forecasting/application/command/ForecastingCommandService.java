package com.metawebthree.forecasting.application.command;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.service.ForecastingDomainService;
import com.metawebthree.forecasting.infrastructure.event.ForecastingEventPublisher;
import org.springframework.stereotype.Service;
import java.time.LocalDate;

@Service
public class ForecastingCommandService {

    private final ForecastingDomainService domainService;
    private final ForecastingEventPublisher eventPublisher;

    public ForecastingCommandService(
            ForecastingDomainService domainService,
            ForecastingEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
    }

    public Long createForecast(String skuCode, String skuName, Long warehouseId,
                               LocalDate forecastDate, Integer quantity, String modelName) {
        SalesForecast forecast = domainService.createForecast(
            skuCode, skuName, warehouseId, forecastDate, quantity, modelName);
        
        eventPublisher.publishForecastCreated(
            forecast.getId(), skuCode, warehouseId, forecastDate, quantity);
        
        return forecast.getId();
    }

    public void confirmForecast(Long forecastId) {
        domainService.confirmForecast(forecastId);
        eventPublisher.publishForecastConfirmed(forecastId);
    }

    public void adjustForecast(Long forecastId, Integer newQuantity) {
        domainService.adjustForecast(forecastId, newQuantity);
        eventPublisher.publishForecastAdjusted(forecastId, newQuantity);
    }

    public void recordActualSales(Long forecastId, Integer actualQuantity) {
        domainService.recordActualSales(forecastId, actualQuantity);
        eventPublisher.publishActualSalesRecorded(forecastId, actualQuantity);
    }

    public Long createModel(String modelName, String modelType, String algorithm,
                           String featureConfig, Integer trainingDays) {
        ForecastModel model = domainService.createModel(
            modelName, modelType, algorithm, featureConfig, trainingDays);
        
        eventPublisher.publishModelCreated(model.getId(), modelName, modelType);
        
        return model.getId();
    }

    public void trainModel(Long modelId) {
        domainService.trainModel(modelId);
        eventPublisher.publishModelTrained(modelId);
    }

    public void deployModel(Long modelId) {
        domainService.deployModel(modelId);
        eventPublisher.publishModelDeployed(modelId);
    }
}