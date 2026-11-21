package com.metawebthree.mes.infrastructure.telemetry;

import com.metawebthree.mes.application.telemetry.TelemetrySourceIngestService;
import com.metawebthree.mes.domain.service.scada.ScadaDomainService;
import com.metawebthree.mes.domain.telemetry.TelemetryMappingProperties;
import com.metawebthree.mes.domain.telemetry.TelemetryNormalizer;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@EnableConfigurationProperties(TelemetryMappingProperties.class)
public class TelemetrySourceConfig {

    @Bean
    public TelemetryNormalizer telemetryNormalizer(TelemetryMappingProperties properties) {
        return new TelemetryNormalizer(properties);
    }

    // Real field path: MQTT source, active when mqtt is enabled.
    @Bean
    @ConditionalOnProperty(prefix = "mqtt", name = "enabled", havingValue = "true")
    public TelemetrySource mqttTelemetrySource(MqttTelemetrySubscriber subscriber,
                                               TelemetryNormalizer normalizer,
                                               TelemetryMappingProperties properties) {
        subscriber.setTopicFilters(properties.getTopicPatterns());
        return new MqttTelemetrySource(subscriber, normalizer, properties.getSourceName());
    }

    // Simulated PLC source: demo/test only, never enabled by default. Produces
    // the same normalized points as a real OPC UA / S7 adapter.
    @Bean(name = "simulatedPlcTelemetrySource")
    @ConditionalOnProperty(prefix = "mes.telemetry.simulated", name = "enabled", havingValue = "true")
    public TelemetrySource simulatedPlcTelemetrySource(TelemetryMappingProperties properties) {
        return new SimulatedPlcTelemetrySource(
            properties.getSimulated().getDeviceCode(),
            properties.getSimulated().getTags(),
            properties.getSimulated().getIntervalMs());
    }

    @Bean
    @ConditionalOnExpression("${mqtt.enabled:false} or ${mes.telemetry.simulated.enabled:false}")
    public TelemetrySourceIngestService telemetrySourceIngestService(
            List<TelemetrySource> sources, ScadaDomainService scadaDomainService) {
        return new TelemetrySourceIngestService(sources, scadaDomainService);
    }
}
