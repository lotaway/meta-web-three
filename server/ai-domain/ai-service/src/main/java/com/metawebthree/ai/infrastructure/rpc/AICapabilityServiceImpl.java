package com.metawebthree.ai.infrastructure.rpc;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

import com.metawebthree.common.generated.rpc.AICapabilityRequest;
import com.metawebthree.common.generated.rpc.AICapabilityResponse;
import com.metawebthree.common.generated.rpc.AICapabilityService;

/**
 * Dubbo provider that makes the existing ai-service the single internal AI
 * capability outlet. Internal callers (factory-service, ai-warehouse-service)
 * reach AI capabilities via {@link AICapabilityService} over Dubbo instead of
 * hard-coded HTTP endpoints.
 *
 * <p>The outbound calls from ai-service to the concrete AI/LLM runtimes are
 * calls to EXTERNAL systems (local LLM server or third-party LLM API) and by
 * design stay HTTP — that is outside the service-to-service call rule. Their
 * endpoints are configurable here (previously hard-coded in callers as
 * localhost:8082/8083/8084) so no caller embeds a URL any more.</p>
 */
@DubboService
public class AICapabilityServiceImpl implements AICapabilityService {

    private static final Logger log = LoggerFactory.getLogger(AICapabilityServiceImpl.class);

    private final RestClient restClient;

    @Value("${ai.capabilities.forecasting.base-url:http://localhost:8082}")
    private String forecastingBaseUrl;
    @Value("${ai.capabilities.location.base-url:http://localhost:8083}")
    private String locationBaseUrl;
    @Value("${ai.capabilities.anomaly.base-url:http://localhost:8084}")
    private String anomalyBaseUrl;
    @Value("${ai.capabilities.recommendation.base-url:http://localhost:8085}")
    private String recommendationBaseUrl;
    @Value("${ai.capabilities.risk.base-url:http://localhost:8086}")
    private String riskBaseUrl;

    public AICapabilityServiceImpl() {
        this.restClient = RestClient.builder()
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    @Override
    public AICapabilityResponse invoke(AICapabilityRequest request) {
        String capability = request.getCapability();
        String payload = request.getPayload();
        String endpoint = resolveEndpoint(capability);

        if (endpoint == null) {
            String error = "Unsupported AI capability: " + capability;
            log.warn(error);
            return AICapabilityResponse.newBuilder()
                    .setSuccess(false).setError(error).build();
        }

        try {
            String body = restClient.post()
                    .uri(endpoint)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(payload)
                    .retrieve()
                    .body(String.class);
            if (body == null) {
                body = "";
            }
            return AICapabilityResponse.newBuilder()
                    .setSuccess(true)
                    .setData(body)
                    .build();
        } catch (Exception e) {
            log.error("AI capability {} invocation failed: {}", capability, e.getMessage());
            return AICapabilityResponse.newBuilder()
                    .setSuccess(false)
                    .setError(e.getMessage() == null ? e.toString() : e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<AICapabilityResponse> invokeAsync(AICapabilityRequest request) {
        return CompletableFuture.completedFuture(invoke(request));
    }

    /**
     * Resolve the configured external AI endpoint for a capability, keeping the
     * path layout used by the former direct HTTP clients.
     */
    private String resolveEndpoint(String capability) {
        if (capability == null) {
            return null;
        }
        return switch (capability) {
            // demand forecasting (factory ForecastingServiceClient + warehouse)
            case "DEMAND_FORECASTING" -> url(forecastingBaseUrl, "/api/forecasting/forecast");
            case "DEMAND_FORECASTING_HISTORY" -> url(forecastingBaseUrl, "/api/forecasting/forecast/history");
            case "DEMAND_FORECASTING_WAREHOUSE" -> url(forecastingBaseUrl, "/api/forecasting/forecast/warehouse");
            case "FORECASTING" -> url(forecastingBaseUrl, "/api/v1/forecast");
            // anomaly detection (factory AnomalyDetectionClient + warehouse)
            case "ANOMALY_DETECTION" -> url(anomalyBaseUrl, "/api/anomaly/detect");
            case "SENSOR_ANOMALY_DETECTION" -> url(anomalyBaseUrl, "/api/anomaly/detect/sensor");
            case "INVENTORY_ANOMALY_DETECTION" -> url(anomalyBaseUrl, "/api/anomaly/detect/inventory");
            case "ACTIVE_ANOMALY_ALERTS" -> url(anomalyBaseUrl, "/api/anomaly/alerts/active");
            case "ANOMALY_PREDICTION" -> url(anomalyBaseUrl, "/api/anomaly/predict");
            case "RISK_SCORING" -> url(riskBaseUrl, "/api/v1/score");
            // location / recommendation (factory LocationRecommendationClient + warehouse)
            case "LOCATION_RECOMMENDATION" -> url(locationBaseUrl, "/api/recommendation/location");
            case "LOCATION_BATCH_RECOMMENDATION" -> url(locationBaseUrl, "/api/recommendation/location/batch");
            case "PRODUCT_CORRELATION" -> url(locationBaseUrl, "/api/recommendation/correlation");
            case "TURNOVER_ANALYSIS" -> url(locationBaseUrl, "/api/recommendation/turnover");
            case "RECOMMENDATION" -> url(recommendationBaseUrl, "/api/v1/recommend");
            default -> fallbackEndpoint(capability);
        };
    }

    /**
     * Warehouse capabilities use generic capability ids; map them to the same
     * external runtimes. Unknown ids are rejected rather than silently routed.
     */
    private String fallbackEndpoint(String capability) {
        return switch (capability) {
            case "demand-forecasting", "ai-forecasting" -> url(forecastingBaseUrl, "/api/v1/forecast");
            case "location-recommendation", "ai-location" -> url(locationBaseUrl, "/api/v1/recommend");
            case "restock-suggestion", "ai-restock" -> url(recommendationBaseUrl, "/api/v1/recommend");
            case "anomaly-detection", "ai-anomaly" -> url(anomalyBaseUrl, "/api/v1/score");
            default -> null;
        };
    }

    private String url(String baseUrl, String path) {
        if (baseUrl == null || baseUrl.isBlank()) {
            return null;
        }
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        return base + path;
    }
}
