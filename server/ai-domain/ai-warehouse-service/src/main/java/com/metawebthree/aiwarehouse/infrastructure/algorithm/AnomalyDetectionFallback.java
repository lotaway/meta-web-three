package com.metawebthree.aiwarehouse.infrastructure.algorithm;

import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter.AlgorithmFallback;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class AnomalyDetectionFallback implements AlgorithmFallback {

    private static final double SIGMA_THRESHOLD = 3.0;

    @Override
    public WarehouseCapability getCapability() {
        return WarehouseCapability.ANOMALY_DETECTION;
    }

    @Override
    public Object execute(String payload) {
        List<Double> sensorData = parseSensorData(payload);
        
        if (sensorData == null || sensorData.size() < 5) {
            return buildAnomalyResult(false, "Insufficient data for detection", null);
        }
        
        double currentValue = getCurrentValue(sensorData);
        DoubleSummaryStatistics stats = sensorData.stream()
            .mapToDouble(Double::doubleValue)
            .summaryStatistics();
        
        double mean = stats.getAverage();
        double stdDev = calculateStdDev(sensorData, mean);
        double lowerBound = mean - SIGMA_THRESHOLD * stdDev;
        double upperBound = mean + SIGMA_THRESHOLD * stdDev;
        
        boolean isAnomaly = currentValue < lowerBound || currentValue > upperBound;
        String anomalyType = determineAnomalyType(currentValue, mean, stdDev);
        
        return buildAnomalyResult(isAnomaly, anomalyType,
            new double[]{lowerBound, upperBound});
    }

    private List<Double> parseSensorData(String payload) {
        if (payload == null || payload.isEmpty()) {
            return null;
        }
        try {
            String[] parts = payload.replaceAll("[^0-9.,-]", "").split(",");
            return Arrays.stream(parts)
                .filter(s -> !s.isEmpty() && !s.equals("."))
                .map(Double::parseDouble)
                .collect(Collectors.toList());
        } catch (Exception e) {
            return null;
        }
    }

    private double getCurrentValue(List<Double> data) {
        return data.get(data.size() - 1);
    }

    private double calculateStdDev(List<Double> data, double mean) {
        double variance = data.stream()
            .mapToDouble(d -> Math.pow(d - mean, 2))
            .average()
            .orElse(0.0);
        return Math.sqrt(variance);
    }

    private String determineAnomalyType(double current, double mean, double stdDev) {
        double zScore = (current - mean) / stdDev;
        if (zScore > SIGMA_THRESHOLD) {
            return "SPIKE";
        } else if (zScore < -SIGMA_THRESHOLD) {
            return "DROP";
        } else {
            return "NORMAL";
        }
    }

    private Object buildAnomalyResult(boolean isAnomaly, String anomalyType,
            double[] bounds) {
        if (bounds != null) {
            return String.format(
                "{\"isAnomaly\":%b,\"anomalyType\":\"%s\",\"lowerBound\":%.2f,"
                + "\"upperBound\":%.2f,\"method\":\"3sigma_threshold\",\"confidence\":0.85}",
                isAnomaly, anomalyType, bounds[0], bounds[1]
            );
        }
        return String.format(
            "{\"isAnomaly\":%b,\"anomalyType\":\"%s\",\"method\":\"3sigma_threshold\"}",
            isAnomaly, anomalyType
        );
    }
}