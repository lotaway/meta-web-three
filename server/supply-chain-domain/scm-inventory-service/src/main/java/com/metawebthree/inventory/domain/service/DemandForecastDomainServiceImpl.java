package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.DemandForecast;
import com.metawebthree.inventory.domain.entity.SalesHistory;
import com.metawebthree.inventory.infrastructure.persistence.repository.DemandForecastRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.SalesHistoryRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class DemandForecastDomainServiceImpl implements DemandForecastDomainService {

    private final DemandForecastRepository forecastRepository;
    private final SalesHistoryRepository salesHistoryRepository;

    public DemandForecastDomainServiceImpl(
            DemandForecastRepository forecastRepository,
            SalesHistoryRepository salesHistoryRepository) {
        this.forecastRepository = forecastRepository;
        this.salesHistoryRepository = salesHistoryRepository;
    }

    @Override
    public DemandForecast generateForecast(String skuCode, Long warehouseId, 
            Integer forecastDays, String method) {
        if (forecastDays == null || forecastDays <= 0) {
            forecastDays = 30;
        }
        if (method == null || method.isEmpty()) {
            method = "SMA";
        }
        
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(90);
        
        List<SalesHistory> salesHistoryList = salesHistoryRepository
                .findBySkuAndWarehouseAndDateRange(skuCode, warehouseId, startDate, endDate);
        
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            throw new IllegalArgumentException("No sales history data available for SKU: " + skuCode);
        }
        
        List<Integer> salesData = salesHistoryList.stream()
                .map(SalesHistory::getQuantity)
                .collect(Collectors.toList());
        
        Integer predictedQuantity;
        int confidenceLevel;
        
        switch (method.toUpperCase()) {
            case "WMA":
                predictedQuantity = calculateWeightedMovingAverage(salesData, null);
                confidenceLevel = 75;
                break;
            case "EXPONENTIAL_SMOOTHING":
                predictedQuantity = calculateExponentialSmoothing(salesData);
                confidenceLevel = 70;
                break;
            case "SMA":
            default:
                predictedQuantity = calculateSimpleMovingAverage(salesData, 7);
                confidenceLevel = 80;
                break;
        }
        
        DemandForecast forecast = new DemandForecast();
        forecast.setSkuCode(skuCode);
        forecast.setWarehouseId(warehouseId);
        forecast.setForecastPeriodDays(forecastDays);
        forecast.setPredictedQuantity(predictedQuantity);
        forecast.setConfidenceLevel(confidenceLevel);
        forecast.setForecastMethod(method);
        forecast.setForecastStartDate(endDate.plusDays(1));
        forecast.setForecastEndDate(endDate.plusDays(forecastDays));
        forecast.setStatus("PENDING");
        forecast.setGeneratedAt(LocalDateTime.now());
        
        return forecastRepository.save(forecast);
    }

    @Override
    public List<DemandForecast> generateForecastsForWarehouse(Long warehouseId, 
            Integer forecastDays, String method) {
        return List.of();
    }

    @Override
    public Integer calculateSimpleMovingAverage(List<Integer> salesData, Integer windowSize) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        if (windowSize == null || windowSize <= 0) {
            windowSize = 7;
        }
        if (windowSize > salesData.size()) {
            windowSize = salesData.size();
        }
        
        int sum = 0;
        int count = 0;
        for (int i = salesData.size() - windowSize; i < salesData.size(); i++) {
            if (salesData.get(i) != null) {
                sum += salesData.get(i);
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0;
    }

    @Override
    public Integer calculateWeightedMovingAverage(List<Integer> salesData, List<Integer> weights) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        int windowSize = 7;
        if (weights != null && weights.size() == windowSize) {
            int sum = 0;
            int weightSum = 0;
            int startIdx = salesData.size() - windowSize;
            for (int i = 0; i < windowSize; i++) {
                int idx = startIdx + i;
                if (idx >= 0 && idx < salesData.size() && salesData.get(idx) != null) {
                    sum += salesData.get(idx) * weights.get(i);
                    weightSum += weights.get(i);
                }
            }
            return weightSum > 0 ? sum / weightSum : 0;
        } else {
            List<Integer> defaultWeights = new ArrayList<>();
            for (int i = 1; i <= windowSize; i++) {
                defaultWeights.add(i);
            }
            return calculateWeightedMovingAverage(salesData, defaultWeights);
        }
    }
    
    private Integer calculateExponentialSmoothing(List<Integer> salesData) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        double alpha = 0.3;
        Integer forecast = salesData.get(0);
        
        for (int i = 1; i < salesData.size(); i++) {
            if (salesData.get(i) != null) {
                forecast = (int) (alpha * salesData.get(i) + (1 - alpha) * forecast);
            }
        }
        
        return forecast;
    }

    @Override
    public List<DemandForecast> getPendingForecasts() {
        return forecastRepository.findByStatus("PENDING");
    }

    @Override
    public DemandForecast approveForecast(Long forecastId) {
        DemandForecast forecast = forecastRepository.findById(forecastId)
                .orElseThrow(() -> new IllegalArgumentException("Demand forecast not found: " + forecastId));
        forecast.setStatus("APPROVED");
        return forecastRepository.save(forecast);
    }

    @Override
    public DemandForecast rejectForecast(Long forecastId) {
        DemandForecast forecast = forecastRepository.findById(forecastId)
                .orElseThrow(() -> new IllegalArgumentException("Demand forecast not found: " + forecastId));
        forecast.setStatus("REJECTED");
        return forecastRepository.save(forecast);
    }
}