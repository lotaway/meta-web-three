package com.metawebthree.digitaltwin.application.ai;

import com.metawebthree.digitaltwin.infrastructure.client.AnomalyDetectionClient;
import com.metawebthree.digitaltwin.infrastructure.client.ForecastingServiceClient;
import com.metawebthree.digitaltwin.infrastructure.client.LocationRecommendationClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;


@Service
public class WarehouseAIService {

    private static final Logger log = LoggerFactory.getLogger(WarehouseAIService.class);

    private final ForecastingServiceClient forecastingClient;
    private final LocationRecommendationClient locationClient;
    private final AnomalyDetectionClient anomalyClient;

    public WarehouseAIService(
            ForecastingServiceClient forecastingClient,
            LocationRecommendationClient locationClient,
            AnomalyDetectionClient anomalyClient) {
        this.forecastingClient = forecastingClient;
        this.locationClient = locationClient;
        this.anomalyClient = anomalyClient;
    }

    // ==================== Demand Forecasting ====================

    
    public ForecastingResult forecastDemand(String skuCode, String skuName,
            Long warehouseId, LocalDate forecastDate) {
        log.info("Requesting AI demand forecast for SKU {} in warehouse {}",
            skuCode, warehouseId);

        ForecastingServiceClient.ForecastResult result =
            forecastingClient.createForecast(skuCode, skuName, warehouseId, forecastDate);

        if (result != null) {
            return mapToForecastingResult(result);
        }

        log.warn("AI forecast unavailable, returning null - caller should use algorithm fallback");
        return null;
    }

    
    public Map<String, Object> getForecastHistory(String skuCode,
            LocalDate startDate, LocalDate endDate) {
        return forecastingClient.getForecastHistory(skuCode, startDate, endDate);
    }

    
    public Map<String, Object> getWarehouseForecasts(Long warehouseId) {
        return forecastingClient.getForecastByWarehouse(warehouseId);
    }

    // ==================== Location Recommendation ====================

    
    public LocationRecResult recommendLocation(String skuCode, Long warehouseId, Integer quantity) {
        log.info("Requesting AI location recommendation for SKU {} in warehouse {}",
            skuCode, warehouseId);

        LocationRecommendationClient.LocationRecommendation result =
            locationClient.recommendLocation(skuCode, warehouseId, quantity);

        if (result != null) {
            return mapToLocationResult(result);
        }

        log.warn("AI location recommendation unavailable, returning null");
        return null;
    }

    
    public List<LocationRecResult> recommendBatchLocations(
            List<Map<String, Object>> items, Long warehouseId) {
        List<LocationRecommendationClient.LocationRecommendation> results =
            locationClient.recommendBatchLocations(items, warehouseId);

        return results.stream()
            .map(this::mapToLocationResult)
            .toList();
    }

    
    public Map<String, Double> getProductCorrelations(String skuCode, Long warehouseId) {
        return locationClient.getProductCorrelations(skuCode, warehouseId);
    }

    
    public Map<String, Object> getTurnoverAnalysis(String skuCode, Long warehouseId) {
        return locationClient.getTurnoverAnalysis(skuCode, warehouseId);
    }

    // ==================== Anomaly Detection ====================

    
    public List<AnomalyResult> detectAnomalies(String skuCode, Long warehouseId, Integer timeRangeHours) {
        log.info("Requesting AI anomaly detection for SKU {} in warehouse {}",
            skuCode, warehouseId);

        List<AnomalyDetectionClient.AnomalyResult> results =
            anomalyClient.detectAnomalies(skuCode, warehouseId, timeRangeHours);

        return results.stream()
            .map(this::mapToAnomalyResult)
            .toList();
    }

    
    public List<AnomalyResult> detectSensorAnomalies(List<Map<String, Object>> sensorData) {
        List<AnomalyDetectionClient.AnomalyResult> results =
            anomalyClient.detectSensorAnomalies(sensorData);

        return results.stream()
            .map(this::mapToAnomalyResult)
            .toList();
    }

    
    public List<AnomalyResult> detectInventoryAnomalies(Long warehouseId,
            LocalDateTime startTime, LocalDateTime endTime) {
        List<AnomalyDetectionClient.AnomalyResult> results =
            anomalyClient.detectInventoryAnomalies(warehouseId, startTime, endTime);

        return results.stream()
            .map(this::mapToAnomalyResult)
            .toList();
    }

    
    public List<AnomalyResult> getActiveAlerts(Long warehouseId, String minSeverity) {
        List<AnomalyDetectionClient.AnomalyResult> results =
            anomalyClient.getActiveAlerts(warehouseId, minSeverity);

        return results.stream()
            .map(this::mapToAnomalyResult)
            .toList();
    }

    
    public List<AnomalyResult> predictAnomalies(String skuCode, Long warehouseId,
            Integer predictionHours) {
        List<AnomalyDetectionClient.AnomalyResult> results =
            anomalyClient.predictAnomalies(skuCode, warehouseId, predictionHours);

        return results.stream()
            .map(this::mapToAnomalyResult)
            .toList();
    }

    // ==================== Mappers ====================

    private ForecastingResult mapToForecastingResult(ForecastingServiceClient.ForecastResult result) {
        ForecastingResult mapped = new ForecastingResult();
        mapped.setForecastId(result.getForecastId());
        mapped.setSkuCode(result.getSkuCode());
        mapped.setPredictedQuantity(result.getPredictedQuantity());
        mapped.setConfidence(result.getConfidence());
        mapped.setModelName(result.getModelName());
        return mapped;
    }

    private LocationRecResult mapToLocationResult(LocationRecommendationClient.LocationRecommendation result) {
        LocationRecResult mapped = new LocationRecResult();
        mapped.setSkuCode(result.getSkuCode());
        mapped.setRecommendedZone(result.getRecommendedZone());
        mapped.setRecommendedShelf(result.getRecommendedShelf());
        mapped.setRecommendedBin(result.getRecommendedBin());
        mapped.setScore(result.getScore());
        mapped.setReason(result.getReason());
        mapped.setAlternativeLocations(result.getAlternativeLocations());
        return mapped;
    }

    private AnomalyResult mapToAnomalyResult(AnomalyDetectionClient.AnomalyResult result) {
        AnomalyResult mapped = new AnomalyResult();
        mapped.setAnomalyId(result.getAnomalyId());
        mapped.setAnomalyType(result.getAnomalyType());
        mapped.setSeverity(result.getSeverity());
        mapped.setDescription(result.getDescription());
        mapped.setTimestamp(result.getTimestamp());
        mapped.setSkuCode(result.getSkuCode());
        mapped.setWarehouseId(result.getWarehouseId());
        mapped.setConfidence(result.getConfidence());
        mapped.setRecommendedAction(result.getRecommendedAction());
        return mapped;
    }

    // ==================== Result DTOs ====================

    public static class ForecastingResult {
        private Long forecastId;
        private String skuCode;
        private Integer predictedQuantity;
        private Double confidence;
        private String modelName;

        public Long getForecastId() {
            return forecastId;
        }

        public void setForecastId(Long forecastId) {
            this.forecastId = forecastId;
        }

        public String getSkuCode() {
            return skuCode;
        }

        public void setSkuCode(String skuCode) {
            this.skuCode = skuCode;
        }

        public Integer getPredictedQuantity() {
            return predictedQuantity;
        }

        public void setPredictedQuantity(Integer predictedQuantity) {
            this.predictedQuantity = predictedQuantity;
        }

        public Double getConfidence() {
            return confidence;
        }

        public void setConfidence(Double confidence) {
            this.confidence = confidence;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }
    }

    public static class LocationRecResult {
        private String skuCode;
        private String recommendedZone;
        private String recommendedShelf;
        private String recommendedBin;
        private Double score;
        private String reason;
        private List<String> alternativeLocations;

        public String getSkuCode() {
            return skuCode;
        }

        public void setSkuCode(String skuCode) {
            this.skuCode = skuCode;
        }

        public String getRecommendedZone() {
            return recommendedZone;
        }

        public void setRecommendedZone(String recommendedZone) {
            this.recommendedZone = recommendedZone;
        }

        public String getRecommendedShelf() {
            return recommendedShelf;
        }

        public void setRecommendedShelf(String recommendedShelf) {
            this.recommendedShelf = recommendedShelf;
        }

        public String getRecommendedBin() {
            return recommendedBin;
        }

        public void setRecommendedBin(String recommendedBin) {
            this.recommendedBin = recommendedBin;
        }

        public Double getScore() {
            return score;
        }

        public void setScore(Double score) {
            this.score = score;
        }

        public String getReason() {
            return reason;
        }

        public void setReason(String reason) {
            this.reason = reason;
        }

        public List<String> getAlternativeLocations() {
            return alternativeLocations;
        }

        public void setAlternativeLocations(List<String> alternativeLocations) {
            this.alternativeLocations = alternativeLocations;
        }
    }

    public static class AnomalyResult {
        private Long anomalyId;
        private String anomalyType;
        private String severity;
        private String description;
        private LocalDateTime timestamp;
        private String skuCode;
        private Long warehouseId;
        private Double confidence;
        private String recommendedAction;

        public Long getAnomalyId() {
            return anomalyId;
        }

        public void setAnomalyId(Long anomalyId) {
            this.anomalyId = anomalyId;
        }

        public String getAnomalyType() {
            return anomalyType;
        }

        public void setAnomalyType(String anomalyType) {
            this.anomalyType = anomalyType;
        }

        public String getSeverity() {
            return severity;
        }

        public void setSeverity(String severity) {
            this.severity = severity;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public LocalDateTime getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(LocalDateTime timestamp) {
            this.timestamp = timestamp;
        }

        public String getSkuCode() {
            return skuCode;
        }

        public void setSkuCode(String skuCode) {
            this.skuCode = skuCode;
        }

        public Long getWarehouseId() {
            return warehouseId;
        }

        public void setWarehouseId(Long warehouseId) {
            this.warehouseId = warehouseId;
        }

        public Double getConfidence() {
            return confidence;
        }

        public void setConfidence(Double confidence) {
            this.confidence = confidence;
        }

        public String getRecommendedAction() {
            return recommendedAction;
        }

        public void setRecommendedAction(String recommendedAction) {
            this.recommendedAction = recommendedAction;
        }
    }
}