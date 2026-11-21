package com.metawebthree.mes.infrastructure.mqtt;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class MqttTelemetryService {

    private final MqttCommandPublisher publisher;

    public MqttTelemetryService(MqttCommandPublisher publisher) {
        this.publisher = publisher;
    }

    @PostConstruct
    public void init() {
        publisher.connect();
    }

    @PreDestroy
    public void shutdown() {
        publisher.disconnect();
    }

    public boolean sendCommand(String equipmentCode, String payload) {
        return publisher.publish(equipmentCode, payload);
    }
}
