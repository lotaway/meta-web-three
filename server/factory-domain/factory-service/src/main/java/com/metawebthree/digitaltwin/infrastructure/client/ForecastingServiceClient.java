package com.metawebthree.digitaltwin.infrastructure.client;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;


@Component
public class ForecastingServiceClient extends AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(ForecastingServiceClient.class);

    public ForecastingServiceClient() {
        super();
    }


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