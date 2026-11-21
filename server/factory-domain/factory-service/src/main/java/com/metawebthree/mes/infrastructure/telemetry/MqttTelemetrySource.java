package com.metawebthree.mes.infrastructure.telemetry;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.telemetry.TelemetryNormalizer;
import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.domain.telemetry.TelemetrySourceException;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Slf4j
public class MqttTelemetrySource implements TelemetrySource {

    private final MqttTelemetrySubscriber subscriber;
    private final TelemetryNormalizer normalizer;
    private final String sourceName;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private Consumer<TelemetryPoint> consumer;

    public MqttTelemetrySource(MqttTelemetrySubscriber subscriber,
                               TelemetryNormalizer normalizer,
                               String sourceName) {
        this.subscriber = subscriber;
        this.normalizer = normalizer;
        this.sourceName = sourceName;
    }

    @Override
    public String name() {
        return sourceName;
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
        try {
            Map<String, Object> data = objectMapper.readValue(payload, Map.class);
            for (TelemetryPoint point : normalizer.normalize(topic, data)) {
                if (consumer != null) consumer.accept(point);
            }
        } catch (Exception e) {
            log.error("Failed to normalize MQTT telemetry from {}: {}", topic, e.getMessage());
        }
    }
}
