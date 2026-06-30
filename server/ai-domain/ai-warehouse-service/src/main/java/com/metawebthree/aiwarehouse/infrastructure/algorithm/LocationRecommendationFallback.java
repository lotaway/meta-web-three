package com.metawebthree.aiwarehouse.infrastructure.algorithm;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter.AlgorithmFallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class LocationRecommendationFallback implements AlgorithmFallback {

    private static final Logger log = LoggerFactory.getLogger(LocationRecommendationFallback.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    private static final double DEFAULT_VELOCITY = 50.0;
    private static final double HIGH_VELOCITY_THRESHOLD = 80.0;
    private static final double MEDIUM_VELOCITY_THRESHOLD = 50.0;
    private static final String ZONE_A = "A";
    private static final String ZONE_B = "B";
    private static final String ZONE_C = "C";
    private static final String ADJACENT_CATEGORY = "RELATED";
    private static final String DEFAULT_CATEGORY = "GENERAL";
    private static final String CLASSIFICATION_METHOD_ABC = "abc_classification";

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
        if (payload == null || payload.isEmpty()) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(payload, 
                new TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            log.warn("Failed to parse payload, returning empty map: {}", e.getMessage());
            return Map.of();
        }
    }

    private String getCategory(Map<String, Object> request) {
        Object category = request.get("category");
        return category != null ? category.toString() : DEFAULT_CATEGORY;
    }

    private double getVelocity(Map<String, Object> request) {
        Object velocity = request.get("velocity");
        if (velocity != null) {
            try {
                return Double.parseDouble(velocity.toString());
            } catch (Exception e) {
                log.warn("Failed to parse velocity, returning default {}: {}", DEFAULT_VELOCITY, e.getMessage());
                return DEFAULT_VELOCITY;
            }
        }
        return DEFAULT_VELOCITY;
    }

    private String assignZone(double velocity) {
        if (velocity >= HIGH_VELOCITY_THRESHOLD) {
            return ZONE_A;
        } else if (velocity >= MEDIUM_VELOCITY_THRESHOLD) {
            return ZONE_B;
        } else {
            return ZONE_C;
        }
    }

    private Object buildRecommendationResult(String zone, String category,
            double velocity) {
        return String.format(
            "{\"recommendedZone\":\"%s\",\"category\":\"%s\",\"velocity\":%.1f,"
            + "\"method\":\"" + CLASSIFICATION_METHOD_ABC + "\",\"adjacentCategories\":[\"%s\"]}",
            zone, category, velocity, findAdjacentCategory(category)
        );
    }

    private String findAdjacentCategory(String category) {
        return ADJACENT_CATEGORY;
    }
}