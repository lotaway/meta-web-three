package com.metawebthree.forecasting.domain.service;

import com.metawebthree.forecasting.domain.entity.SalesForecast;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesForecastRepository;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    private static final int DEFAULT_HISTORY_DAYS = 90;
    private static final int MIN_TRAINING_DAYS = 7;
    private static final int TRAIN_SPLIT_PERCENT = 75;
    private static final double DEFAULT_ALPHA = 0.3;
    private static final int EFFECTIVENESS_THRESHOLD = 80;
    private static final int ACCURACY_WEIGHT = 70;
    private static final int RANKING_LIMIT = 10;
    private static final int PARAM_PENALTY_THRESHOLD = 95;
    private static final int PARAM_PENALTY_FLOOR = 60;
    private static final int DEFAULT_TRAINING_DAYS = 30;
    private static final int DEFAULT_WMA_ACCURACY = 72;
    private static final int DEFAULT_SMA_ACCURACY = 78;
    private static final int DEPLOY_THRESHOLD_PERCENT = 70;
    private static final String DEFAULT_ALGORITHM = "SMA";

    private static final Logger log = LoggerFactory.getLogger(ForecastingDomainServiceImpl.class);

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
        forecastRepository.save(forecast);
        return forecast;
    }

    @Override
    public SalesForecast createForecastWithAlgorithm(String skuCode, String skuName,
            Long warehouseId, LocalDate forecastDate, String algorithm, Integer windowSize) {
        
        List<Integer> salesData = getSalesHistoryData(skuCode, warehouseId);
        
        if (windowSize == null || windowSize <= 0) {
            windowSize = MIN_TRAINING_DAYS;
        }
        
        Integer predictedQuantity = predictByAlgorithm(salesData, algorithm, windowSize);
        BigDecimal accuracy = calculateModelAccuracy(salesData, algorithm, windowSize);
        
        SalesForecast forecast = new SalesForecast();
        forecast.create(skuCode, skuName, warehouseId, forecastDate, predictedQuantity,
                        algorithm, accuracy);
        forecastRepository.save(forecast);
        return forecast;
    }

    private List<Integer> getSalesHistoryData(String skuCode, Long warehouseId) {
        List<SalesHistory> salesHistoryList = salesHistoryRepository
                .findRecentBySkuCodeAndWarehouseId(skuCode, warehouseId, DEFAULT_HISTORY_DAYS);
        
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            throw new IllegalArgumentException("No sales history data available for SKU: " + skuCode);
        }
        
        salesHistoryList.sort((a, b) -> a.getSalesDate().compareTo(b.getSalesDate()));
        
        return salesHistoryList.stream()
                .map(SalesHistory::getQuantity)
                .collect(Collectors.toList());
    }

    private Integer predictByAlgorithm(List<Integer> salesData, String algorithm, Integer windowSize) {
        switch (algorithm != null ? algorithm.toUpperCase() : DEFAULT_ALGORITHM) {
            case "WMA":
                return calculateWeightedMovingAverage(salesData, windowSize);
            case "EXPONENTIAL_SMOOTHING":
                return calculateExponentialSmoothing(salesData);
            case "SMA":
            default:
                return calculateSimpleMovingAverage(salesData, windowSize);
        }
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
        modelRepository.save(model);
        return model;
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
            trainingDays = DEFAULT_TRAINING_DAYS;
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
            
            Long warehouseId = parseWarehouseId(pair);
            if (warehouseId == null) continue;
            
            BigDecimal accuracy = processPairAccuracy(skuCode.trim(), warehouseId, algorithm, trainingDays);
            if (accuracy != null && accuracy.compareTo(BigDecimal.ZERO) > 0) {
                totalAccuracy += accuracy.doubleValue();
                validScopes++;
            }
        }
        
        if (validScopes == 0) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        double avgAccuracy = totalAccuracy / validScopes;
        return BigDecimal.valueOf(clampAccuracy(avgAccuracy))
                .setScale(2, RoundingMode.HALF_UP);
    }

    private Long parseWarehouseId(String[] pair) {
        if (pair.length > 1 && pair[1] != null && !pair[1].trim().isEmpty()) {
            try {
                return Long.parseLong(pair[1].trim());
            } catch (NumberFormatException e) {
                log.warn("Invalid number format", e);
            }
        }
        return null;
    }

    private BigDecimal processPairAccuracy(String skuCode, Long warehouseId, String algorithm, Integer trainingDays) {
        List<SalesHistory> history = salesHistoryRepository
            .findRecentBySkuCodeAndWarehouseId(skuCode, warehouseId, trainingDays);
        
        if (history == null || history.size() < RANKING_LIMIT) return null;
        
        history.sort((a, b) -> a.getSalesDate().compareTo(b.getSalesDate()));
        
        return calculateTrainingAccuracy(history, algorithm);
    }

    private double clampAccuracy(double accuracy) {
        return Math.min(PARAM_PENALTY_THRESHOLD, Math.max(PARAM_PENALTY_FLOOR, accuracy));
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

    public Integer calculateSimpleMovingAverage(List<Integer> salesData, Integer windowSize) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        if (windowSize == null || windowSize <= 0) {
            windowSize = MIN_TRAINING_DAYS;
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

    public Integer calculateWeightedMovingAverage(List<Integer> salesData, Integer windowSize) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        if (windowSize == null || windowSize <= 0) {
            windowSize = MIN_TRAINING_DAYS;
        }
        if (windowSize > salesData.size()) {
            windowSize = salesData.size();
        }
        
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

    public Integer calculateExponentialSmoothing(List<Integer> salesData) {
        if (salesData == null || salesData.isEmpty()) {
            return 0;
        }
        
        double alpha = DEFAULT_ALPHA;
        double forecast = salesData.get(0);
        
        for (int i = 1; i < salesData.size(); i++) {
            if (salesData.get(i) != null) {
                forecast = alpha * salesData.get(i) + (1 - alpha) * forecast;
            }
        }
        
        return (int) Math.round(forecast);
    }

    private BigDecimal calculateModelAccuracy(List<Integer> salesData, String algorithm, Integer windowSize) {
        if (salesData == null || salesData.size() < windowSize + 1) {
            return BigDecimal.valueOf(ACCURACY_WEIGHT);
        }
        
        List<Integer> trainingData = salesData.subList(0, salesData.size() - 1);
        List<Integer> testData = salesData.subList(salesData.size() - 1, salesData.size());
        
        Integer predicted;
        switch (algorithm != null ? algorithm.toUpperCase() : DEFAULT_ALGORITHM) {
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
            return BigDecimal.valueOf(ACCURACY_WEIGHT);
        }
        
        double error = Math.abs(predicted - actual) / (double) actual;
        double accuracy = Math.max(0, (1 - error) * 100);
        
        return BigDecimal.valueOf(accuracy).setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calculateTrainingAccuracy(List<SalesHistory> salesHistoryList, String algorithm) {
        if (salesHistoryList == null || salesHistoryList.isEmpty()) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        List<Integer> salesData = salesHistoryList.stream()
                .map(SalesHistory::getQuantity)
                .collect(Collectors.toList());
        
        if (salesData.size() < RANKING_LIMIT) {
            return calculateDefaultAccuracy(algorithm);
        }
        
        int testSize = Math.min(MIN_TRAINING_DAYS, salesData.size() / 3);
        List<Integer> trainingData = salesData.subList(0, salesData.size() - testSize);
        List<Integer> testData = salesData.subList(salesData.size() - testSize, salesData.size());
        
        double totalAccuracy = 0;
        int validPredictions = 0;
        
        for (int i = 0; i < testData.size(); i++) {
            Integer predicted = predictForWindow(trainingData, testData, i, algorithm);
            
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
        return BigDecimal.valueOf(clampAccuracy(avgAccuracy))
                .setScale(2, RoundingMode.HALF_UP);
    }

    private Integer predictForWindow(List<Integer> trainingData, List<Integer> testData, int i, String algorithm) {
        int windowStart = trainingData.size() - MIN_TRAINING_DAYS + i;
        if (windowStart < 0) windowStart = 0;
        List<Integer> window = new ArrayList<>(trainingData.subList(windowStart, trainingData.size()));
        window.addAll(testData.subList(0, i));
        
        switch (algorithm != null ? algorithm.toUpperCase() : DEFAULT_ALGORITHM) {
            case "WMA":
                return calculateWeightedMovingAverage(window, MIN_TRAINING_DAYS);
            case "EXPONENTIAL_SMOOTHING":
                return calculateExponentialSmoothing(window);
            case "SMA":
            default:
                return calculateSimpleMovingAverage(window, MIN_TRAINING_DAYS);
        }
    }

    private BigDecimal calculateDefaultAccuracy(String algorithm) {
        switch (algorithm != null ? algorithm.toUpperCase() : DEFAULT_ALGORITHM) {
            case "WMA":
                return BigDecimal.valueOf(TRAIN_SPLIT_PERCENT);
            case "EXPONENTIAL_SMOOTHING":
                return BigDecimal.valueOf(DEFAULT_WMA_ACCURACY);
            case "SMA":
            default:
                return BigDecimal.valueOf(DEFAULT_SMA_ACCURACY);
        }
    }

    @Override
    public void deployModel(Long modelId) {
        ForecastModel model = modelRepository.findById(modelId)
            .orElseThrow(() -> new IllegalArgumentException("Model not found"));
        
        if (!model.isReadyForDeployment()) {
            throw new IllegalStateException("Model accuracy below threshold (" + DEPLOY_THRESHOLD_PERCENT + "%)");
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