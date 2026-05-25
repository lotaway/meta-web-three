package com.metawebthree.digitaltwin.infrastructure.client;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpResponse;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Client for AI-based anomaly detection.
 * Uses time-series sensor data + inventory changes to detect anomalies.
 */
@Component
public class AnomalyDetectionClient extends AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(AnomalyDetectionClient.class);

    public AnomalyDetectionClient() {
        super("http://localhost:8084", 5000, 3);
    }

    /**
     * Detect anomalies for a specific SKU in a warehouse.
     *
     * @param skuCode Product SKU code
     * @param warehouseId Warehouse ID
     * @param timeRange Hours to analyze
     * @return List of detected anomalies
     */
    public List<AnomalyResult> detectAnomalies(String skuCode, Long warehouseId, Integer timeRange) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("warehouseId", warehouseId);
        payload.put("timeRangeHours", timeRange);
        payload.put("detectionType", "COMBINED");

        AIClientRequest request = new AIClientRequest("ANOMALY_DETECTION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseAnomalyResults(response.getData());
        }

        log.error("Failed to detect anomalies for SKU {}: {}", skuCode, response.getError());
        return List.of();
    }

    /**
     * Detect anomalies based on sensor time-series data.
     *
     * @param sensorData List of sensor readings with timestamp
     * @return Detected anomalies in the sensor data
     */
    public List<AnomalyResult> detectSensorAnomalies(List<Map<String, Object>> sensorData) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("sensorData", sensorData);
        payload.put("detectionType", "SENSOR_TIME_SERIES");

        AIClientRequest request = new AIClientRequest("SENSOR_ANOMALY_DETECTION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseAnomalyResults(response.getData());
        }

        log.error("Failed to detect sensor anomalies: {}", response.getError());
        return List.of();
    }

    /**
     * Detect anomalies based on inventory change patterns.
     *
     * @param warehouseId Warehouse ID
     * @param startTime Start of analysis window
     * @param endTime End of analysis window
     * @return List of inventory pattern anomalies
     */
    public List<AnomalyResult> detectInventoryAnomalies(Long warehouseId,
            LocalDateTime startTime, LocalDateTime endTime) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("warehouseId", warehouseId);
        payload.put("startTime", startTime.toString());
        payload.put("endTime", endTime.toString());
        payload.put("detectionType", "INVENTORY_PATTERN");

        AIClientRequest request = new AIClientRequest("INVENTORY_ANOMALY_DETECTION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseAnomalyResults(response.getData());
        }

        log.error("Failed to detect inventory anomalies for warehouse {}: {}",
            warehouseId, response.getError());
        return List.of();
    }

    /**
     * Get real-time anomaly alerts for a warehouse.
     *
     * @param warehouseId Warehouse ID
     * @param alertSeverity Minimum severity level (LOW, MEDIUM, HIGH, CRITICAL)
     * @return List of active anomaly alerts
     */
    public List<AnomalyResult> getActiveAlerts(Long warehouseId, String alertSeverity) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("warehouseId", warehouseId);
        payload.put("minSeverity", alertSeverity);

        AIClientRequest request = new AIClientRequest("ACTIVE_ANOMALY_ALERTS", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseAnomalyResults(response.getData());
        }

        log.error("Failed to get active anomaly alerts for warehouse {}: {}",
            warehouseId, response.getError());
        return List.of();
    }

    /**
     * Get anomaly prediction for a SKU (predict future anomalies).
     *
     * @param skuCode Product SKU code
     * @param warehouseId Warehouse ID
     * @param predictionHours Hours to predict ahead
     * @return Predicted anomalies
     */
    public List<AnomalyResult> predictAnomalies(String skuCode, Long warehouseId,
            Integer predictionHours) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("warehouseId", warehouseId);
        payload.put("predictionHours", predictionHours);

        AIClientRequest request = new AIClientRequest("ANOMALY_PREDICTION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseAnomalyResults(response.getData());
        }

        log.error("Failed to predict anomalies for SKU {}: {}", skuCode, response.getError());
        return List.of();
    }

    @Override
    protected AIClientResponse doInvoke(AIClientRequest request) {
        try {
            String path = getApiPath(request.getCapability());
            URI uri = URI.create(endpoint + path);
            String body = request.toJson();

            HttpRequest httpRequest = buildRequest(uri, body).build();
            HttpResponse<String> httpResponse = httpClient.send(httpRequest,
                HttpResponse.BodyHandlers.ofString());

            if (httpResponse.statusCode() >= 200 && httpResponse.statusCode() < 300) {
                log.debug("Anomaly detection request succeeded: {}", request.getCapability());
                return AIClientResponse.success(httpResponse.body(), httpResponse.statusCode());
            } else {
                log.warn("Anomaly detection request failed with status {}: {}",
                    httpResponse.statusCode(), httpResponse.body());
                return AIClientResponse.failure(
                    "HTTP " + httpResponse.statusCode() + ": " + httpResponse.body(),
                    httpResponse.statusCode()
                );
            }
        } catch (Exception e) {
            log.error("Anomaly detection request exception: {}", e.getMessage());
            return AIClientResponse.failure(e.getMessage(), -1);
        }
    }

    private String getApiPath(String capability) {
        return switch (capability) {
            case "ANOMALY_DETECTION" -> "/api/anomaly/detect";
            case "SENSOR_ANOMALY_DETECTION" -> "/api/anomaly/detect/sensor";
            case "INVENTORY_ANOMALY_DETECTION" -> "/api/anomaly/detect/inventory";
            case "ACTIVE_ANOMALY_ALERTS" -> "/api/anomaly/alerts/active";
            case "ANOMALY_PREDICTION" -> "/api/anomaly/predict";
            default -> "/api/anomaly/detect";
        };
    }

    @SuppressWarnings("unchecked")
    private List<AnomalyResult> parseAnomalyResults(String json) {
        try {
            List<Map<String, Object>> dataList = objectMapper.readValue(json, List.class);
            List<AnomalyResult> results = new ArrayList<>();
            for (Map<String, Object> data : dataList) {
                AnomalyResult result = new AnomalyResult();
                result.setAnomalyId(((Number) data.getOrDefault("anomalyId", 0)).longValue());
                result.setAnomalyType((String) data.get("anomalyType"));
                result.setSeverity((String) data.get("severity"));
                result.setDescription((String) data.get("description"));
                result.setTimestamp(LocalDateTime.parse((String) data.get("timestamp")));
                result.setSkuCode((String) data.get("skuCode"));
                result.setWarehouseId(((Number) data.get("warehouseId")).longValue());
                result.setConfidence(((Number) data.getOrDefault("confidence", 0.0)).doubleValue());
                result.setRecommendedAction((String) data.get("recommendedAction"));
                results.add(result);
            }
            return results;
        } catch (Exception e) {
            log.error("Failed to parse anomaly results: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * Anomaly detection result DTO.
     */
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
        private Map<String, Object> metadata;

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

        public Map<String, Object> getMetadata() {
            return metadata;
        }

        public void setMetadata(Map<String, Object> metadata) {
            this.metadata = metadata;
        }
    }
}