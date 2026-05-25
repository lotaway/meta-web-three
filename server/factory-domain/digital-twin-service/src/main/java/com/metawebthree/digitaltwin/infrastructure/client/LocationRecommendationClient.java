package com.metawebthree.digitaltwin.infrastructure.client;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpResponse;
import java.util.*;

/**
 * Client for AI-based location recommendation.
 * Uses product correlation + turnover rate to recommend optimal storage locations.
 */
@Component
public class LocationRecommendationClient extends AbstractAIClient {

    private static final Logger log = LoggerFactory.getLogger(LocationRecommendationClient.class);

    public LocationRecommendationClient() {
        super("http://localhost:8083", 5000, 3);
    }

    /**
     * Recommend optimal storage location for a product.
     *
     * @param skuCode Product SKU code
     * @param warehouseId Warehouse ID
     * @param quantity Quantity to store
     * @return Recommended location with score
     */
    public LocationRecommendation recommendLocation(String skuCode, Long warehouseId, Integer quantity) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("warehouseId", warehouseId);
        payload.put("quantity", quantity);
        payload.put("strategy", "CORRELATION_TURNOVER");

        AIClientRequest request = new AIClientRequest("LOCATION_RECOMMENDATION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseRecommendation(response.getData());
        }

        log.error("Failed to get location recommendation for SKU {}: {}",
            skuCode, response.getError());
        return null;
    }

    /**
     * Recommend locations for multiple products (batch optimization).
     *
     * @param items List of SKU and quantity pairs
     * @param warehouseId Warehouse ID
     * @return List of recommendations with overall optimization score
     */
    public List<LocationRecommendation> recommendBatchLocations(
            List<Map<String, Object>> items, Long warehouseId) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("items", items);
        payload.put("warehouseId", warehouseId);
        payload.put("strategy", "BATCH_OPTIMIZATION");

        AIClientRequest request = new AIClientRequest("LOCATION_BATCH_RECOMMENDATION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return parseBatchRecommendations(response.getData());
        }

        log.error("Failed to get batch location recommendations: {}", response.getError());
        return List.of();
    }

    /**
     * Get product correlation data for location planning.
     *
     * @param skuCode Product SKU code
     * @param warehouseId Warehouse ID
     * @return Map of correlated products and their correlation scores
     */
    public Map<String, Double> getProductCorrelations(String skuCode, Long warehouseId) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("warehouseId", warehouseId);

        AIClientRequest request = new AIClientRequest("PRODUCT_CORRELATION", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return response.getDataAsMap();
        }

        log.error("Failed to get product correlations for SKU {}: {}",
            skuCode, response.getError());
        return Map.of();
    }

    /**
     * Get turnover rate analysis for a product.
     *
     * @param skuCode Product SKU code
     * @param warehouseId Warehouse ID
     * @return Turnover analysis data
     */
    public Map<String, Object> getTurnoverAnalysis(String skuCode, Long warehouseId) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("skuCode", skuCode);
        payload.put("warehouseId", warehouseId);

        AIClientRequest request = new AIClientRequest("TURNOVER_ANALYSIS", payload);
        AIClientResponse response = invoke(request);

        if (response.isSuccess()) {
            return response.getDataAsMap();
        }

        log.error("Failed to get turnover analysis for SKU {}: {}",
            skuCode, response.getError());
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
                log.debug("Location recommendation request succeeded: {}", request.getCapability());
                return AIClientResponse.success(httpResponse.body(), httpResponse.statusCode());
            } else {
                log.warn("Location recommendation request failed with status {}: {}",
                    httpResponse.statusCode(), httpResponse.body());
                return AIClientResponse.failure(
                    "HTTP " + httpResponse.statusCode() + ": " + httpResponse.body(),
                    httpResponse.statusCode()
                );
            }
        } catch (Exception e) {
            log.error("Location recommendation request exception: {}", e.getMessage());
            return AIClientResponse.failure(e.getMessage(), -1);
        }
    }

    private String getApiPath(String capability) {
        return switch (capability) {
            case "LOCATION_RECOMMENDATION" -> "/api/recommendation/location";
            case "LOCATION_BATCH_RECOMMENDATION" -> "/api/recommendation/location/batch";
            case "PRODUCT_CORRELATION" -> "/api/recommendation/correlation";
            case "TURNOVER_ANALYSIS" -> "/api/recommendation/turnover";
            default -> "/api/recommendation/location";
        };
    }

    private LocationRecommendation parseRecommendation(String json) {
        try {
            Map<String, Object> data = objectMapper.readValue(json, Map.class);
            LocationRecommendation result = new LocationRecommendation();
            result.setSkuCode((String) data.get("skuCode"));
            result.setRecommendedZone((String) data.get("recommendedZone"));
            result.setRecommendedShelf((String) data.get("recommendedShelf"));
            result.setRecommendedBin((String) data.get("recommendedBin"));
            result.setScore(((Number) data.getOrDefault("score", 0.0)).doubleValue());
            result.setReason((String) data.get("reason"));
            result.setAlternativeLocations(parseAlternatives(data.get("alternativeLocations")));
            return result;
        } catch (Exception e) {
            log.error("Failed to parse location recommendation: {}", e.getMessage());
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private List<LocationRecommendation> parseBatchRecommendations(String json) {
        try {
            List<Map<String, Object>> dataList = objectMapper.readValue(json, List.class);
            List<LocationRecommendation> results = new ArrayList<>();
            for (Map<String, Object> data : dataList) {
                LocationRecommendation rec = new LocationRecommendation();
                rec.setSkuCode((String) data.get("skuCode"));
                rec.setRecommendedZone((String) data.get("recommendedZone"));
                rec.setRecommendedShelf((String) data.get("recommendedShelf"));
                rec.setScore(((Number) data.getOrDefault("score", 0.0)).doubleValue());
                results.add(rec);
            }
            return results;
        } catch (Exception e) {
            log.error("Failed to parse batch recommendations: {}", e.getMessage());
            return List.of();
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> parseAlternatives(Object alternatives) {
        if (alternatives == null) {
            return List.of();
        }
        try {
            return (List<String>) alternatives;
        } catch (Exception e) {
            return List.of();
        }
    }

    /**
     * Location recommendation result DTO.
     */
    public static class LocationRecommendation {
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
}