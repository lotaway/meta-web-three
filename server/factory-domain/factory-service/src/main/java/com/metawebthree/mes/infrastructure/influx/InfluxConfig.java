package com.metawebthree.mes.infrastructure.influx;

import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.InfluxDBClientFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "meta.influx")
@ConditionalOnProperty(prefix = "meta.influx", name = "enabled", havingValue = "true")
public class InfluxConfig {

    private String url = "http://localhost:8086";
    private String token = "";
    private String org = "meta";
    private String bucket = "telemetry";

    @Bean
    public InfluxDBClient influxDBClient() {
        return InfluxDBClientFactory.create(url, token.toCharArray(), org, bucket);
    }

    @Bean
    public InfluxTelemetryStore influxTelemetryStore(InfluxDBClient client) {
        return new InfluxTelemetryStore(client, bucket);
    }
}
