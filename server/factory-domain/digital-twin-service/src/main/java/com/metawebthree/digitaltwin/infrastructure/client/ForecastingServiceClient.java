package com.metawebthree.digitaltwin.infrastructure.client;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpResponse;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

/**
 * Client for connecting to forecasting-service.
 * Provides AI-based demand forecasting by calling the forecasting-service REST API.
 */
@Component
public class ForecastingServiceClient extends AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(ForecastingServiceClient.class);

    @Value("${ai.forecasting-service.endpoint:http://localhost:8082}")
    private String defaultEndpoint;

    @Value("${ai.forecasting-service.timeout-ms:5000}")
    private int defaultTimeout;

    @Value("${ai.forecasting-service.retry-count:3}")
    private int defaultRetryCount;

    public ForecastingServiceClient() {
        super("http://localhost:8082", 5000, 3);
    }

    /**
     * Create a forecast request for a specific SKU.
     *
     * @param skuCode Product SKU code
     * @param skuName Product name
     * @param warehouseId Warehouse ID
     * @param forecastDate Date to forecast
     * @return Forecast result with predicted quantity and confidence
     */
    public ForecastResult createForecast(String skuCode, String skuName,
            Long warehouseId, LocalDate forecastDate) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("skuName", skuName);
        payload.put("warehouseId", warehouseId);
        payload.put("forecastDate", forecastDate.toString());
        payload.put("modelName", "default");

        AIClientRequest request = new AIClientRequest("DEMAND_FORECASTING", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseForecastResult(response.getData());
        }

        log.error("Failed to create forecast for SKU {}: {}", skuCode, response.getError());
        return null;
    }

    /**
     * Get historical forecasts for a SKU within a date range.
     *
     * @param skuCode Product SKU code
     * @param startDate Start date of history
     * @param endDate End date of history
     * @return List of historical forecast data
     */
    public Map<String, Object> getForecastHistory(String skuCode,
            LocalDate startDate, LocalDate endDate) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("startDate", startDate.toString());
        payload.put("endDate", endDate.toString());

        AIClientRequest request = new AIClientRequest("DEMAND_FORECASTING_HISTORY", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return response.getDataAsMap();
        }

        log.error("Failed to get forecast history for SKU {}: {}", skuCode, response.getError());
        return Map.of();
    }

    /**
     * Get forecasts by warehouse.
     *
     * @param warehouseId Warehouse ID
     * @return List of forecasts for the warehouse
     */
    public Map<String, Object> getForecastByWarehouse(Long warehouseId) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("warehouseId", warehouseId);

        AIClientRequest request = new AIClientRequest("DEMAND_FORECASTING_WAREHOUSE", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return response.getDataAsMap();
        }

        log.error("Failed to get forecasts for warehouse {}: {}", warehouseId, response.getError());
        return Map.of();
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
                log.debug("Forecast request succeeded: {}", request.getCapability());
                return AIClientResponse.success(httpResponse.body(), httpResponse.statusCode());
            } else {
                log.warn("Forecast request failed with status {}: {}",
                    httpResponse.statusCode(), httpResponse.body());
                return AIClientResponse.failure(
                    "HTTP " + httpResponse.statusCode() + ": " + httpResponse.body(),
                    httpResponse.statusCode()
                );
            }
        } catch (Exception e) {
            log.error("Forecast request exception: {}", e.getMessage());
            return AIClientResponse.failure(e.getMessage(), -1);
        }
    }

    private String getApiPath(String capability) {
        return switch (capability) {
            case "DEMAND_FORECASTING" -> "/api/forecasting/forecast";
            case "DEMAND_FORECASTING_HISTORY" -> "/api/forecasting/forecast/history";
            case "DEMAND_FORECASTING_WAREHOUSE" -> "/api/forecasting/forecast/warehouse";
            default -> "/api/forecasting/forecast";
        };
    }

    private ForecastResult parseForecastResult(String json) {
        try {
            Map<String, Object> data = objectMapper.readValue(json, Map.class);
            ForecastResult result = new ForecastResult();
            result.setForecastId(((Number) data.get("forecastId")).longValue());
            result.setSkuCode((String) data.get("skuCode"));
            result.setPredictedQuantity(((Number) data.getOrDefault("predictedQuantity", 0)).intValue());
            result.setConfidence(((Number) data.getOrDefault("confidence", 0.0)).doubleValue());
            result.setModelName((String) data.getOrDefault("modelName", "default"));
            return result;
        } catch (Exception e) {
            log.error("Failed to parse forecast result: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Forecast result DTO.
     */
    public static class ForecastResult {
        private Long forecastId;
        private String skuCode;
        private Integer predictedQuantity;
        private Double confidence;
        private String modelName;
        private LocalDate forecastDate;

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

        public LocalDate getForecastDate() {
            return forecastDate;
        }

        public void setForecastDate(LocalDate forecastDate) {
            this.forecastDate = forecastDate;
        }
    }
}