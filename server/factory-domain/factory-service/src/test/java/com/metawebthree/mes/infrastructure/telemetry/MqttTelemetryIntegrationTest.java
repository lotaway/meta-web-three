package com.metawebthree.mes.infrastructure.telemetry;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import com.metawebthree.mes.domain.telemetry.TelemetryMappingProperties;
import com.metawebthree.mes.domain.telemetry.TelemetryNormalizer;
import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import com.metawebthree.mes.infrastructure.mqtt.MqttTelemetrySubscriber;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Real-broker integration test for the MQTT telemetry path.
 *
 * <p>Requires a local broker (docker compose -f docker-compose.env.yml up -d mqtt).
 * Opt-in via MQTT_IT=1 so it never runs in the default unit-test suite. Uses an
 * isolated topic prefix (itest/) so test data never enters a production topic.</p>
 */
@EnabledIfEnvironmentVariable(named = "MQTT_IT", matches = "1")
class MqttTelemetryIntegrationTest {

    private static final String BROKER = System.getenv().getOrDefault("MQTT_IT_BROKER", "tcp://localhost:1883");
    // Isolated prefix: never overlaps the production topic (mes/equipment/).
    private static final String PREFIX = System.getenv().getOrDefault("MQTT_IT_PREFIX", "itest/equipment/");

    @Test
    void receivesAndNormalizesTelemetryFromRealBroker() throws Exception {
        String topic = PREFIX + "IT-DEVICE-1/telemetry";
        TelemetryMappingProperties props = new TelemetryMappingProperties();
        TelemetryNormalizer normalizer = new TelemetryNormalizer(props);
        MqttTelemetrySubscriber subscriber = new MqttTelemetrySubscriber(
                BROKER, "it-sub-" + System.currentTimeMillis(), PREFIX, true);
        MqttTelemetrySource source = new MqttTelemetrySource(subscriber, normalizer, "IT");

        List<TelemetryPoint> received = new CopyOnWriteArrayList<>();
        CountDownLatch latch = new CountDownLatch(1);
        source.subscribe(point -> {
            received.add(point);
            latch.countDown();
        });
        source.connect();
        Thread.sleep(500); // allow SUBACK before publishing

        MqttClient publisher = new MqttClient(BROKER, "it-pub-" + System.currentTimeMillis(), new MemoryPersistence());
        MqttConnectOptions opts = new MqttConnectOptions();
        opts.setAutomaticReconnect(false);
        publisher.connect(opts);
        String payload = "{\"timestamp\":1700000000000,\"quality\":\"good\",\"temperature\":23.5,\"unit\":{\"temperature\":\"C\"}}";
        publisher.publish(topic, new MqttMessage(payload.getBytes(StandardCharsets.UTF_8)));

        boolean delivered = latch.await(10, TimeUnit.SECONDS);
        publisher.disconnect();
        publisher.close();
        source.close();

        assertThat(delivered).as("telemetry point received from broker").isTrue();
        assertThat(received).hasSize(1);
        TelemetryPoint point = received.get(0);
        assertThat(point.equipmentCode()).isEqualTo("IT-DEVICE-1");
        assertThat(point.pointKey()).isEqualTo("temperature");
        assertThat(point.value()).isEqualTo(23.5);
        assertThat(point.quality()).isEqualTo(TelemetryPoint.QUALITY_GOOD);
        assertThat(point.timestamp().toEpochMilli()).isEqualTo(1700000000000L);
    }
}
