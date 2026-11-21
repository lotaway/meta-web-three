package com.metawebthree.mes.infrastructure.telemetry;

import com.metawebthree.mes.application.telemetry.TelemetrySourceIngestService;
import com.metawebthree.mes.domain.service.scada.ScadaDomainService;
import com.metawebthree.mes.domain.telemetry.TelemetryMappingProperties;
import com.metawebthree.mes.domain.telemetry.TelemetryNormalizer;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(TelemetryMappingProperties.class)
@ConditionalOnProperty(prefix = "mqtt", name = "enabled", havingValue = "true")
public class TelemetrySourceConfig {

    @Bean
    public TelemetryNormalizer telemetryNormalizer(TelemetryMappingProperties properties) {
        return new TelemetryNormalizer(properties);
    }

    @Bean
    public TelemetrySource mqttTelemetrySource(MqttTelemetrySubscriber subscriber,
                                               TelemetryNormalizer normalizer,
                                               TelemetryMappingProperties properties) {
        subscriber.setTopicFilters(properties.getTopicPatterns());
        return new MqttTelemetrySource(subscriber, normalizer, properties.getSourceName());
    }

    @Bean
    public TelemetrySourceIngestService telemetrySourceIngestService(
            TelemetrySource source, ScadaDomainService scadaDomainService) {
        return new TelemetrySourceIngestService(source, scadaDomainService);
    }
}
