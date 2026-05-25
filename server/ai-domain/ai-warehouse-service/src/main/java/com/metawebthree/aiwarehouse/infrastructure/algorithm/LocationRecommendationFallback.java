package com.metawebthree.aiwarehouse.infrastructure.algorithm;

import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter.AlgorithmFallback;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
public class LocationRecommendationFallback implements AlgorithmFallback {

    @Override
    public WarehouseCapability getCapability() {
        return WarehouseCapability.LOCATION_RECOMMENDATION;
    }

    @Override
    public Object execute(String payload) {
        Map<String, Object> request = parsePayload(payload);
        String productCategory = getCategory(request);
        double productVelocity = getVelocity(request);
        String zone = assignZone(productVelocity);
        
        return buildRecommendationResult(zone, productCategory, productVelocity);
    }

    private Map<String, Object> parsePayload(String payload) {
        return Map.of();
    }

    private String getCategory(Map<String, Object> request) {
        Object category = request.get("category");
        return category != null ? category.toString() : "GENERAL";
    }

    private double getVelocity(Map<String, Object> request) {
        Object velocity = request.get("velocity");
        if (velocity != null) {
            try {
                return Double.parseDouble(velocity.toString());
            } catch (Exception e) {
                return 50.0;
            }
        }
        return 50.0;
    }

    private String assignZone(double velocity) {
        if (velocity >= 80) {
            return "A";
        } else if (velocity >= 50) {
            return "B";
        } else {
            return "C";
        }
    }

    private Object buildRecommendationResult(String zone, String category,
            double velocity) {
        return String.format(
            "{\"recommendedZone\":\"%s\",\"category\":\"%s\",\"velocity\":%.1f,"
            + "\"method\":\"abc_classification\",\"adjacentCategories\":[\"%s\"]}",
            zone, category, velocity, findAdjacentCategory(category)
        );
    }

    private String findAdjacentCategory(String category) {
        return "RELATED";
    }
}