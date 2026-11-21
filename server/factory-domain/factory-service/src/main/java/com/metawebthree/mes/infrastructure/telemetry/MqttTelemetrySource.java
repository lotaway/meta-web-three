package com.metawebthree.mes.infrastructure.telemetry;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.domain.telemetry.TelemetrySourceException;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;
import lombok.extern.slf4j.Slf4j;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Slf4j
public class MqttTelemetrySource implements TelemetrySource {

    private final MqttTelemetrySubscriber subscriber;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private Consumer<TelemetryPoint> consumer;

    public MqttTelemetrySource(MqttTelemetrySubscriber subscriber) {
        this.subscriber = subscriber;
    }

    @Override
    public String name() {
        return "MQTT";
    }

    @Override
    public void connect() throws TelemetrySourceException {
        subscriber.connect();
    }

    @Override
    public void subscribe(Consumer<TelemetryPoint> consumer) throws TelemetrySourceException {
        this.consumer = consumer;
        subscriber.setMessageHandler(this::handleMessage);
    }

    @Override
    public List<TelemetryPoint> read() throws TelemetrySourceException {
        throw new TelemetrySourceException("MQTT source is subscription-only");
    }

    @Override
    public void close() {
        subscriber.disconnect();
    }

    private void handleMessage(String topic, String payload) {
        String equipmentCode = extractEquipmentCode(topic);
        if (equipmentCode == null) return;
        try {
            Map<String, Object> data = objectMapper.readValue(payload, Map.class);
            Instant timestamp = extractTimestamp(data);
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                if (entry.getValue() instanceof Number) {
                    emit(new TelemetryPoint(equipmentCode, entry.getKey(),
                        ((Number) entry.getValue()).doubleValue(), null,
                        TelemetryPoint.QUALITY_GOOD, timestamp));
                }
            }
        } catch (Exception e) {
            log.error("Failed to normalize MQTT telemetry from {}: {}", topic, e.getMessage());
        }
    }

    private void emit(TelemetryPoint point) {
        if (consumer != null) consumer.accept(point);
    }

    private Instant extractTimestamp(Map<String, Object> data) {
        Object ts = data.get("timestamp");
        if (ts instanceof Number) return Instant.ofEpochMilli(((Number) ts).longValue());
        return Instant.now();
    }

    private String extractEquipmentCode(String topic) {
        if (topic == null) return null;
        String[] parts = topic.split("/");
        return parts.length >= 3 ? parts[parts.length - 2] : null;
    }
}
