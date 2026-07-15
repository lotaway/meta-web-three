package com.metawebthree.mes.infrastructure.mqtt;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.paho.client.mqttv3.*;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;

import java.util.function.BiConsumer;

@Slf4j
public class MqttTelemetrySubscriber {

    private final String brokerUrl;
    private final String clientId;
    private final String topicPrefix;
    private final boolean enabled;
    private MqttClient client;
    private BiConsumer<String, String> messageHandler;

    public MqttTelemetrySubscriber(String brokerUrl, String clientId, String topicPrefix, boolean enabled) {
        this.brokerUrl = brokerUrl;
        this.clientId = clientId;
        this.topicPrefix = topicPrefix;
        this.enabled = enabled;
    }

    public void setMessageHandler(BiConsumer<String, String> handler) {
        this.messageHandler = handler;
    }

    public void connect() {
        if (!enabled) {
            log.info("MQTT subscriber is disabled");
            return;
        }
        try {
            client = new MqttClient(brokerUrl, clientId, new MemoryPersistence());
            MqttConnectOptions opts = new MqttConnectOptions();
            opts.setCleanSession(true);
            opts.setAutomaticReconnect(true);
            opts.setConnectionTimeout(30);
            opts.setKeepAliveInterval(60);
            client.connect(opts);
            String topicFilter = topicPrefix + "+/telemetry";
            client.subscribe(topicFilter, this::handleMessage);
            log.info("MQTT subscriber connected to {} subscribed to {}", brokerUrl, topicFilter);
        } catch (MqttException e) {
            log.error("MQTT subscriber connection failed: {}", e);
        }
    }

    private void handleMessage(String topic, MqttMessage message) {
        String payload = new String(message.getPayload());
        log.debug("MQTT received from {}: {}", topic, payload);
        if (messageHandler != null) {
            messageHandler.accept(topic, payload);
        }
    }

    public void disconnect() {
        if (client != null && client.isConnected()) {
            try {
                client.disconnect();
                client.close();
            } catch (MqttException e) {
                log.warn("MQTT disconnect error: {}", e.getMessage());
            }
        }
    }
}
