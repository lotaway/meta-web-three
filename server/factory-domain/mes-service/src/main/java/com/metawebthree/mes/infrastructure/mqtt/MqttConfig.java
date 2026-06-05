package com.metawebthree.mes.infrastructure.mqtt;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MqttConfig {

    @Value("${mqtt.broker.url:tcp://localhost:1883}")
    private String brokerUrl;

    @Value("${mqtt.client.id:mes-service-" + "${random.uuid}" + "}")
    private String clientId;

    @Value("${mqtt.topic.prefix:mes/equipment/}")
    private String topicPrefix;

    @Value("${mqtt.enabled:false}")
    private boolean enabled;

    @Bean
    public MqttTelemetrySubscriber mqttTelemetrySubscriber() {
        return new MqttTelemetrySubscriber(brokerUrl, clientId, topicPrefix, enabled);
    }

    @Bean
    public MqttCommandPublisher mqttCommandPublisher() {
        return new MqttCommandPublisher(brokerUrl, clientId + "-cmd", topicPrefix, enabled);
    }
}
