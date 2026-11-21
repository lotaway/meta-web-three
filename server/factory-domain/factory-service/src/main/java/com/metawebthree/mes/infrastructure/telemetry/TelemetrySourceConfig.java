package com.metawebthree.mes.infrastructure.telemetry;

import com.metawebthree.mes.application.telemetry.TelemetrySourceIngestService;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnProperty(prefix = "mqtt", name = "enabled", havingValue = "true")
public class TelemetrySourceConfig {

    @Bean
    public TelemetrySource mqttTelemetrySource(MqttTelemetrySubscriber subscriber) {
        return new MqttTelemetrySource(subscriber);
    }

    @Bean
    public TelemetrySourceIngestService telemetrySourceIngestService(
            TelemetrySource source,
            com.metawebthree.mes.domain.service.scada.ScadaDomainService scadaDomainService) {
        return new TelemetrySourceIngestService(source, scadaDomainService);
    }
}
