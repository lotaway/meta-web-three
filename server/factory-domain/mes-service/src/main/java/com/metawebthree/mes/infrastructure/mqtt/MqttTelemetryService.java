package com.metawebthree.mes.infrastructure.mqtt;

import com.metawebthree.mes.domain.service.scada.ScadaDomainService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class MqttTelemetryService {

    private final MqttTelemetrySubscriber subscriber;
    private final MqttCommandPublisher publisher;
    private final ScadaDomainService scadaDomainService;

    public MqttTelemetryService(MqttTelemetrySubscriber subscriber,
                                 MqttCommandPublisher publisher,
                                 ScadaDomainService scadaDomainService) {
        this.subscriber = subscriber;
        this.publisher = publisher;
        this.scadaDomainService = scadaDomainService;
    }

    @PostConstruct
    public void init() {
        subscriber.setMessageHandler((topic, payload) -> {
            try {
                String equipmentCode = extractEquipmentCode(topic);
                if (equipmentCode != null) {
                    scadaDomainService.ingestTelemetry(equipmentCode, topic, payload, null);
                }
            } catch (Exception e) {
                log.error("Failed to process telemetry from {}: {}", topic, e.getMessage());
            }
        });
        subscriber.connect();
        publisher.connect();
    }

    @PreDestroy
    public void shutdown() {
        subscriber.disconnect();
        publisher.disconnect();
    }

    public boolean sendCommand(String equipmentCode, String payload) {
        return publisher.publish(equipmentCode, payload);
    }

    private String extractEquipmentCode(String topic) {
        if (topic == null) return null;
        String[] parts = topic.split("/");
        if (parts.length >= 3) return parts[parts.length - 2];
        return null;
    }
}
