package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.infrastructure.client.AnomalyDetectionClient;
import com.metawebthree.digitaltwin.infrastructure.client.ForecastingServiceClient;
import com.metawebthree.digitaltwin.infrastructure.client.LocationRecommendationClient;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * AI service client configuration.
 * Configures endpoints and connection parameters for AI/ML services.
 */
@Configuration
public class AIServiceClientConfig {

    @Bean
    @ConfigurationProperties(prefix = "ai.forecasting-service")
    public ForecastingServiceClient forecastingServiceClient() {
        return new ForecastingServiceClient();
    }

    @Bean
    @ConfigurationProperties(prefix = "ai.location-recommendation-service")
    public LocationRecommendationClient locationRecommendationClient() {
        return new LocationRecommendationClient();
    }

    @Bean
    @ConfigurationProperties(prefix = "ai.anomaly-detection-service")
    public AnomalyDetectionClient anomalyDetectionClient() {
        return new AnomalyDetectionClient();
    }
}