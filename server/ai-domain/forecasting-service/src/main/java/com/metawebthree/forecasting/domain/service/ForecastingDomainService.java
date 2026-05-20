package com.metawebthree.forecasting.domain.service;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface ForecastingDomainService {
    SalesForecast createForecast(String skuCode, String skuName, Long warehouseId,
                                  LocalDate forecastDate, Integer quantity, String modelName);
    
    void confirmForecast(Long forecastId);
    
    void adjustForecast(Long forecastId, Integer newQuantity);
    
    void recordActualSales(Long forecastId, Integer actualQuantity);
    
    List<SalesForecast> getForecastHistory(String skuCode, LocalDate startDate, 
                                            LocalDate endDate);
    
    ForecastModel createModel(String modelName, String modelType, String algorithm,
                              String featureConfig, Integer trainingDays);
    
    void trainModel(Long modelId);
    
    void deployModel(Long modelId);
    
    Optional<ForecastModel> getDeployedModel(String modelType);
    
    Double calculateForecastAccuracy(Long forecastId);
}