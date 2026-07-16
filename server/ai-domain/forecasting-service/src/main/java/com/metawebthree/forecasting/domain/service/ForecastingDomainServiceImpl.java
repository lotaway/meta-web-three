package com.metawebthree.forecasting.domain.service;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class ForecastingDomainServiceImpl implements ForecastingDomainService {

    private final SalesForecastRepository forecastRepository;
    private final ForecastModelRepository modelRepository;
    private final SalesHistoryRepository salesHistoryRepository;

    public ForecastingDomainServiceImpl(
            SalesForecastRepository forecastRepository,
            ForecastModelRepository modelRepository,
            SalesHistoryRepository salesHistoryRepository) {
        this.forecastRepository = forecastRepository;
        this.modelRepository = modelRepository;
        this.salesHistoryRepository = salesHistoryRepository;
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
    public SalesForecast createForecastWithAlgorithm(String skuCode, String skuName,
            Long warehouseId, LocalDate forecastDate, String algorithm, Integer windowSize) {
        
        // Get historical sales data (last 90 days)
        List<SalesHistory> salesHistoryList = salesHistoryRepository
                .findRecentBySkuCodeAndWarehouseId(skuCode, warehouseId, 90);
        
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            throw new IllegalArgumentException("No sales history data available for SKU: " + skuCode);
        }
        
        // Sort by date
        salesHistoryList.sort((a, b) -> a.getSalesDate().compareTo(b.getSalesDate()));
        
        // Extract sales quantities
        List<Integer> salesData = salesHistoryList.stream()
                .map(SalesHistory::getQuantity)
                .collect(Collectors.toList());
        
        // Calculate predicted quantity based on algorithm
        Integer predictedQuantity;
        int confidenceLevel;
        
        if (windowSize == null || windowSize <= 0) {
            windowSize = 7;
        }
        
        switch (algorithm != null ? algorithm.toUpperCase() : "SMA") {
            case "WMA":
                predictedQuantity = calculateWeightedMovingAverage(salesData, windowSize);
                confidenceLevel = 75;
                break;
            case "EXPONENTIAL_SMOOTHING":
                predictedQuantity = calculateExponentialSmoothing(salesData);
                confidenceLevel = 70;
                break;
            case "SMA":
            default:
                predictedQuantity = calculateSimpleMovingAverage(salesData, windowSize);
                confidenceLevel = 80;
                break;
        }
        
        // Calculate model accuracy based on historical prediction error
        BigDecimal accuracy = calculateModelAccuracy(salesData, algorithm, windowSize);
        
        SalesForecast forecast = new SalesForecast();
        forecast.create(skuCode, skuName, warehouseId, forecastDate, predictedQuantity,
                        algorithm, accuracy);
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
        
        String algorithm = model.getAlgorithm();
        Integer trainingDays = model.getTrainingDays();
        
        if (trainingDays == null || trainingDays <= 0) {
            trainingDays = 30;
        }
        
        BigDecimal accuracy = computeAccuracyFromTrainingData(
            model.getFeatureConfig(), algorithm, trainingDays);
        
        model.completeTraining(accuracy);
        modelRepository.update(model);
    }

    private BigDecimal computeAccuracyFromTrainingData(
            String featureConfig, String algorithm, Integer trainingDays) {
        
        List<String[]> pairs = parseSkuWarehousePairs(featureConfig);
        
        if (pairs.isEmpty()) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        double totalAccuracy = 0;
        int validScopes = 0;
        
        for (String[] pair : pairs) {
            String skuCode = pair[0];
            if (skuCode == null || skuCode.trim().isEmpty()) continue;
            
            Long warehouseId = null;
            if (pair.length > 1 && pair[1] != null && !pair[1].trim().isEmpty()) {
                try {
                    warehouseId = Long.parseLong(pair[1].trim());
                } catch (NumberFormatException e) {
                    continue;
                }
            }
            
            if (warehouseId == null) continue;
            
            List<SalesHistory> history = salesHistoryRepository
                .findRecentBySkuCodeAndWarehouseId(skuCode.trim(), warehouseId, trainingDays);
            
            if (history == null || history.size() < 10) continue;
            
            history.sort((a, b) -> a.getSalesDate().compareTo(b.getSalesDate()));
            
            BigDecimal accuracy = calculateTrainingAccuracy(history, algorithm);
            if (accuracy != null && accuracy.compareTo(BigDecimal.ZERO) > 0) {
                totalAccuracy += accuracy.doubleValue();
                validScopes++;
            }
        }
        
        if (validScopes == 0) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        double avgAccuracy = totalAccuracy / validScopes;
        return BigDecimal.valueOf(Math.min(95, Math.max(60, avgAccuracy)))
                .setScale(2, RoundingMode.HALF_UP);
    }
    
    private List<String[]> parseSkuWarehousePairs(String featureConfig) {
        List<String[]> pairs = new ArrayList<>();
        if (featureConfig == null || featureConfig.trim().isEmpty()) {
            return pairs;
        }
        String[] entries = featureConfig.split(",");
        for (String entry : entries) {
            String[] parts = entry.trim().split(":");
            if (parts.length >= 1 && !parts[0].trim().isEmpty()) {
                pairs.add(parts);
            }
        }
        return pairs;
    }

    /**
     * Calculate Simple Moving Average (SMA)
     */
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

    /**
     * Calculate Weighted Moving Average (WMA)
     * More recent data gets higher weight
     */
    public Integer calculateWeightedMovingAverage(List<Integer> salesData, Integer windowSize) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        if (windowSize == null || windowSize <= 0) {
            windowSize = 7;
        }
        if (windowSize > salesData.size()) {
            windowSize = salesData.size();
        }
        
        // Default weights: more recent = higher weight
        // For window size 7: weights are [1,2,3,4,5,6,7] where 7 is most recent
        List<Integer> weights = new ArrayList<>();
        for (int i = 1; i <= windowSize; i++) {
            weights.add(i);
        }
        
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
    }

    /**
     * Calculate Exponential Smoothing
     * Gives more weight to recent observations
     */
    public Integer calculateExponentialSmoothing(List<Integer> salesData) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        double alpha = 0.3; // Smoothing factor (0 < alpha < 1)
        double forecast = salesData.get(0);
        
        for (int i = 1; i < salesData.size(); i++) {
            if (salesData.get(i) != null) {
                forecast = alpha * salesData.get(i) + (1 - alpha) * forecast;
            }
        }
        
        return (int) Math.round(forecast);
    }

    /**
     * Calculate model accuracy based on historical prediction errors
     */
    private BigDecimal calculateModelAccuracy(List<Integer> salesData, String algorithm, Integer windowSize) {
        if (salesData == null || salesData.size() < windowSize + 1) {
            return BigDecimal.valueOf(70); // Default accuracy if not enough data
        }
        
        List<Integer> trainingData = salesData.subList(0, salesData.size() - 1);
        List<Integer> testData = salesData.subList(salesData.size() - 1, salesData.size());
        
        Integer predicted;
        switch (algorithm != null ? algorithm.toUpperCase() : "SMA") {
            case "WMA":
                predicted = calculateWeightedMovingAverage(trainingData, windowSize);
                break;
            case "EXPONENTIAL_SMOOTHING":
                predicted = calculateExponentialSmoothing(trainingData);
                break;
            case "SMA":
            default:
                predicted = calculateSimpleMovingAverage(trainingData, windowSize);
                break;
        }
        
        int actual = testData.get(0);
        if (actual == 0) {
            return BigDecimal.valueOf(70);
        }
        
        double error = Math.abs(predicted - actual) / (double) actual;
        double accuracy = Math.max(0, (1 - error) * 100);
        
        return BigDecimal.valueOf(accuracy).setScale(2, RoundingMode.HALF_UP);
    }

    /**
     * Calculate training accuracy from historical data
     */
    private BigDecimal calculateTrainingAccuracy(List<SalesHistory> salesHistoryList, String algorithm) {
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            // Generate simulated accuracy based on algorithm effectiveness
            return calculateDefaultAccuracy(algorithm);
        }
        
        // Group by SKU and warehouse
        // For each group, calculate prediction accuracy
        // This is simplified - in production would use cross-validation
        
        List<Integer> salesData = salesHistoryList.stream()
                .map(SalesHistory::getQuantity)
                .collect(Collectors.toList());
        
        if (salesData.size() < 10) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        // Use hold-out validation: predict last N days from earlier data
        int testSize = Math.min(7, salesData.size() / 3);
        List<Integer> trainingData = salesData.subList(0, salesData.size() - testSize);
        List<Integer> testData = salesData.subList(salesData.size() - testSize, salesData.size());
        
        double totalAccuracy = 0;
        int validPredictions = 0;
        
        for (int i = 0; i < testData.size(); i++) {
            // Get the window of data before the test point
            int windowStart = trainingData.size() - 7 + i;
            if (windowStart < 0) windowStart = 0;
            List<Integer> window = new ArrayList<>(trainingData.subList(windowStart, trainingData.size()));
            window.addAll(testData.subList(0, i));
            
            Integer predicted;
            switch (algorithm != null ? algorithm.toUpperCase() : "SMA") {
                case "WMA":
                    predicted = calculateWeightedMovingAverage(window, 7);
                    break;
                case "EXPONENTIAL_SMOOTHING":
                    predicted = calculateExponentialSmoothing(window);
                    break;
                case "SMA":
                default:
                    predicted = calculateSimpleMovingAverage(window, 7);
                    break;
            }
            
            int actual = testData.get(i);
            if (actual > 0) {
                double accuracy = 1 - Math.abs(predicted - actual) / (double) actual;
                totalAccuracy += Math.max(0, accuracy);
                validPredictions++;
            }
        }
        
        if (validPredictions == 0) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        double avgAccuracy = totalAccuracy / validPredictions * 100;
        return BigDecimal.valueOf(Math.min(95, Math.max(60, avgAccuracy)))
                .setScale(2, RoundingMode.HALF_UP);
    }

    /**
     * Calculate default accuracy based on algorithm type
     */
    private BigDecimal calculateDefaultAccuracy(String algorithm) {
        switch (algorithm != null ? algorithm.toUpperCase() : "SMA") {
            case "WMA":
                return BigDecimal.valueOf(75);
            case "EXPONENTIAL_SMOOTHING":
                return BigDecimal.valueOf(72);
            case "SMA":
            default:
                return BigDecimal.valueOf(78);
        }
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