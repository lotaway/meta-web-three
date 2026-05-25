package com.metawebthree.aiwarehouse.infrastructure.algorithm;

import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter.AlgorithmFallback;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class DemandForecastingFallback implements AlgorithmFallback {

    private static final double ALPHA = 0.3;
    private static final double SAFETY_STOCK_Z = 1.65;
    private static final int DEFAULT_LEAD_TIME = 7;

    @Override
    public WarehouseCapability getCapability() {
        return WarehouseCapability.DEMAND_FORECASTING;
    }

    @Override
    public Object execute(String payload) {
        List<Double> historicalData = parseHistoricalData(payload);
        if (historicalData == null || historicalData.size() < 3) {
            return defaultForecast();
        }
        
        double exponentialSmoothed = exponentialSmoothing(historicalData);
        double movingAverage = movingAverage(historicalData);
        double forecast = (exponentialSmoothed + movingAverage) / 2.0;
        double safetyStock = calculateSafetyStock(historicalData);
        
        return buildForecastResult(forecast, safetyStock);
    }

    private List<Double> parseHistoricalData(String payload) {
        if (payload == null || payload.isEmpty()) {
            return null;
        }
        try {
            String[] parts = payload.replaceAll("[^0-9.,]", "").split(",");
            return Arrays.stream(parts)
                .filter(s -> !s.isEmpty())
                .map(Double::parseDouble)
                .collect(Collectors.toList());
        } catch (Exception e) {
            return null;
        }
    }

    private double exponentialSmoothing(List<Double> data) {
        double level = data.get(0);
        for (int i = 1; i < data.size(); i++) {
            level = ALPHA * data.get(i) + (1 - ALPHA) * level;
        }
        return level;
    }

    private double movingAverage(List<Double> data) {
        int window = Math.min(3, data.size());
        return data.subList(data.size() - window, data.size())
            .stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
    }

    private double calculateSafetyStock(List<Double> data) {
        DoubleSummaryStatistics stats = data.stream()
            .mapToDouble(Double::doubleValue)
            .summaryStatistics();
        
        double avg = stats.getAverage();
        double variance = data.stream()
            .mapToDouble(d -> Math.pow(d - avg, 2))
            .average()
            .orElse(0.0);
        double stdDev = Math.sqrt(variance);
        
        return SAFETY_STOCK_Z * stdDev * Math.sqrt(DEFAULT_LEAD_TIME);
    }

    private Object defaultForecast() {
        return buildForecastResult(0.0, 0.0);
    }

    private Object buildForecastResult(double forecast, double safetyStock) {
        return String.format(
            "{\"forecast\":%.2f,\"safetyStock\":%.2f,\"method\":\"algorithm\",\"confidence\":0.7}",
            forecast, safetyStock
        );
    }
}