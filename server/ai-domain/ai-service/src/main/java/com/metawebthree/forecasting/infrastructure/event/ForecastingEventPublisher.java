package com.metawebthree.forecasting.infrastructure.event;

import com.metawebthree.forecasting.domain.event.ForecastingEventType;
import org.springframework.stereotype.Component;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

@Component
public class ForecastingEventPublisher {

    public void publishForecastCreated(Long forecastId, String skuCode,
            Long warehouseId, LocalDate forecastDate, Integer quantity) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.FORECAST_CREATED);
        event.put("forecastId", forecastId);
        event.put("skuCode", skuCode);
        event.put("warehouseId", warehouseId);
        event.put("forecastDate", forecastDate);
        event.put("quantity", quantity);
    }

    public void publishForecastConfirmed(Long forecastId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.FORECAST_CONFIRMED);
        event.put("forecastId", forecastId);
    }

    public void publishForecastAdjusted(Long forecastId, Integer newQuantity) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.FORECAST_ADJUSTED);
        event.put("forecastId", forecastId);
        event.put("newQuantity", newQuantity);
    }

    public void publishActualSalesRecorded(Long forecastId, Integer actualQuantity) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.ACTUAL_SALES_RECORDED);
        event.put("forecastId", forecastId);
        event.put("actualQuantity", actualQuantity);
    }

    public void publishModelCreated(Long modelId, String modelName, String modelType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.MODEL_CREATED);
        event.put("modelId", modelId);
        event.put("modelName", modelName);
        event.put("modelType", modelType);
    }

    public void publishModelTrained(Long modelId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.MODEL_TRAINED);
        event.put("modelId", modelId);
    }

    public void publishModelDeployed(Long modelId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", ForecastingEventType.MODEL_DEPLOYED);
        event.put("modelId", modelId);
    }
}
