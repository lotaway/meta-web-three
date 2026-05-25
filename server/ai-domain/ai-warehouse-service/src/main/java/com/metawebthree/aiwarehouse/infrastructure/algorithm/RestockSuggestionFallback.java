package com.metawebthree.aiwarehouse.infrastructure.algorithm;

import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter.AlgorithmFallback;
import org.springframework.stereotype.Component;

@Component
public class RestockSuggestionFallback implements AlgorithmFallback {

    private static final int LEAD_TIME_DAYS = 7;
    private static final double SAFETY_FACTOR = 1.2;

    @Override
    public WarehouseCapability getCapability() {
        return WarehouseCapability.RESTOCK_SUGGESTION;
    }

    @Override
    public Object execute(String payload) {
        double currentStock = parseCurrentStock(payload);
        double dailyConsumption = parseDailyConsumption(payload);
        double reorderPoint = calculateReorderPoint(dailyConsumption);
        double suggestedQuantity = calculateSuggestedQuantity(
            currentStock, dailyConsumption, reorderPoint);
        
        return buildSuggestionResult(currentStock, dailyConsumption,
            reorderPoint, suggestedQuantity);
    }

    private double parseCurrentStock(String payload) {
        if (payload == null || payload.isEmpty()) {
            return 0.0;
        }
        try {
            String stock = extractValue(payload, "currentStock", "stock", "quantity");
            return stock != null ? Double.parseDouble(stock) : 0.0;
        } catch (Exception e) {
            return 0.0;
        }
    }

    private double parseDailyConsumption(String payload) {
        if (payload == null || payload.isEmpty()) {
            return 10.0;
        }
        try {
            String consumption = extractValue(payload, "dailyConsumption",
                "consumption", "demand");
            return consumption != null ? Double.parseDouble(consumption) : 10.0;
        } catch (Exception e) {
            return 10.0;
        }
    }

    private String extractValue(String payload, String... keys) {
        for (String key : keys) {
            int idx = payload.indexOf("\"" + key + "\"");
            if (idx >= 0) {
                int colon = payload.indexOf(":", idx);
                int comma = payload.indexOf(",", colon);
                int end = comma > 0 ? comma : payload.length();
                String value = payload.substring(colon + 1, end).trim();
                value = value.replaceAll("[^0-9.-]", "");
                if (!value.isEmpty()) {
                    return value;
                }
            }
        }
        return null;
    }

    private double calculateReorderPoint(double dailyConsumption) {
        return dailyConsumption * LEAD_TIME_DAYS * SAFETY_FACTOR;
    }

    private double calculateSuggestedQuantity(double currentStock,
            double dailyConsumption, double reorderPoint) {
        double maxStock = dailyConsumption * LEAD_TIME_DAYS * 2 * SAFETY_FACTOR;
        double suggested = maxStock - currentStock;
        return Math.max(0, suggested);
    }

    private Object buildSuggestionResult(double currentStock, double dailyConsumption,
            double reorderPoint, double suggestedQuantity) {
        boolean isUrgent = currentStock < reorderPoint * 0.5;
        return String.format(
            "{\"currentStock\":%.0f,\"dailyConsumption\":%.1f,\"reorderPoint\":%.1f,"
            + "\"suggestedQuantity\":%.0f,\"urgency\":\"%s\",\"leadTimeDays\":%d,"
            + "\"method\":\"consumption_prediction\"}",
            currentStock, dailyConsumption, reorderPoint, suggestedQuantity,
            isUrgent ? "HIGH" : "NORMAL", LEAD_TIME_DAYS
        );
    }
}